from collections import namedtuple
from typing import Dict, Union

import torch
from torch import nn as nn

from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoModel


class ColBERTerConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    dropout: float = 0.0
    return_vecs: bool = False
    dual_loss: bool = False
    trainable: bool = True

    compression_dim: int = -1
    use_contextualized_stopwords = False
    aggregate_unique_ids = False
    retrieval_compression_dim = -1
    
    compress_to_exact_mini_mode = False
    second_compress_dim = -1

class ColBERTer(PreTrainedModel):
    """
    ColBERTer model
    """

    config_class = ColBERTerConfig
    base_model_prefix = "bert_model"
    #is_teacher_model = False  # gets overriden by the dynamic teacher runner

    @staticmethod
    def from_config(config):
        cfg = ColBERTerConfig()
        cfg.bert_model = config["bert_pretrained_model"]
        cfg.return_vecs = config.get("in_batch_negatives", False)
        cfg.dual_loss = config.get("train_dual_loss", False)
        cfg.trainable = config["bert_trainable"]

        cfg.compression_dim = config["colberter_compression_dim"]
        cfg.retrieval_compression_dim = config["colberter_retrieval_compression_dim"]
        cfg.use_contextualized_stopwords = config["colberter_use_contextualized_stopwords"]
        cfg.aggregate_unique_ids = config["colberter_aggregate_unique_ids"]
        cfg.compress_to_exact_mini_mode = config.get("colberter_compress_to_exact_mini_mode", False)
        cfg.second_compress_dim = config.get("colberter_second_compress_dim", -1)

        return ColBERTer(cfg)

    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)

        self.return_vecs = cfg.return_vecs
        self.dual_loss = cfg.dual_loss

        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)

        self.score_merger = nn.Parameter(torch.zeros(1))

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self.use_compressor = cfg.compression_dim > -1
        if self.use_compressor:
            self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)

        self.use_retrieval_compression = cfg.retrieval_compression_dim > -1
        if self.use_retrieval_compression:
            self.compressor_retrieval = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.retrieval_compression_dim)

        self.use_contextualized_stopwords = cfg.use_contextualized_stopwords
        if self.use_contextualized_stopwords:
            self.stop_word_reducer = nn.Linear(cfg.compression_dim, 1, bias=True)
            torch.nn.init.constant_(self.stop_word_reducer.bias, 1)  # make sure we don't start in a broken state

        self.aggregate_unique_ids = cfg.aggregate_unique_ids

        self.compress_to_exact_mini_mode = cfg.compress_to_exact_mini_mode
        self.use_second_compression = cfg.second_compress_dim > -1
        if self.use_second_compression:
            self.mini_compressor = torch.nn.Linear(cfg.compression_dim, cfg.second_compress_dim)


    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16: bool = True,
                output_secondary_output: bool = False) -> torch.Tensor:

        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_retrieval_vec, query_vecs, query_mask, _ = self.forward_representation(query, sequence_type="query_encode_internal")
            document_retrieval_vec, document_vecs, document_mask, document_stop_words = self.forward_representation(document, sequence_type="doc_encode_internal")

            score_retrieval = torch.bmm(query_retrieval_vec.unsqueeze(dim=1),
                                        document_retrieval_vec.unsqueeze(dim=2)).squeeze(-1).squeeze(-1)

            exact_scoring_mask=None
            if self.compress_to_exact_mini_mode:
                if self.aggregate_unique_ids:
                    exact_scoring_mask = query["unique_words"].unsqueeze(-1) == document["unique_words"].unsqueeze(1)
                else:
                    exact_scoring_mask = query["input_ids"].unsqueeze(-1) == document["input_ids"].unsqueeze(1)

            score, score_refine, score_per_term = self.forward_aggregation(score_retrieval, query_vecs, query_mask, document_vecs, document_mask,return_all_scores=True,exact_scoring_mask=exact_scoring_mask)

            #if self.is_teacher_model:
            #    # return (score, score_per_term)
            #    return (score, query_vecs, document_vecs)

            # used for in-batch negatives, we return them for multi-gpu sync -> out of the forward() method
            if self.training and self.return_vecs and self.dual_loss:
                score = (score, score_retrieval, query_retrieval_vec, document_retrieval_vec)
            elif self.training and self.dual_loss:
                score = (score, score_retrieval)

            if output_secondary_output:
                secondary_dict = {"score_per_term": score_per_term}
                if self.aggregate_unique_ids:
                    # document
                    secondary_dict["document_all_ids"] = document["input_ids"]
                    secondary_dict["document_unique_words"] = document["unique_words"]
                    secondary_dict["document_input_ids_to_words_map"] = document["input_ids_to_words_map"]
                    # query
                    secondary_dict["query_all_ids"] = query["input_ids"]
                    secondary_dict["query_unique_words"] = query["unique_words"]
                    secondary_dict["query_input_ids_to_words_map"] = query["input_ids_to_words_map"]
                if self.use_contextualized_stopwords:
                    document_stop_words[~(document_mask)] = -1
                    secondary_dict["document_stop_words"] = document_stop_words
                return score, secondary_dict

            if self.use_contextualized_stopwords:
                return score, [document_stop_words[document_mask]]
            else:
                return score

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type=None):

        stopword_importance = None  # if we are not using stopwords (disabled, or for the query)

        if sequence_type == "doc_encode" or sequence_type == "doc_encode_internal":

            cls_vec, token_vecs, token_mask = self.forward_shared_encoding(tokens)

            #
            # Part 2, Refinement 2.3: learn stopwords
            #
            if self.use_contextualized_stopwords:
                stopword_importance = torch.nn.functional.relu(self.stop_word_reducer(token_vecs))
                token_vecs = token_vecs * stopword_importance

        if sequence_type == "query_encode" or sequence_type == "query_encode_internal":

            cls_vec, token_vecs, token_mask = self.forward_shared_encoding(tokens)

        if self.use_second_compression:
            token_vecs = self.mini_compressor(token_vecs)

        # only mask for inference, because we'll mask the scores later anyway for internal scoring
        if sequence_type == "doc_encode" or sequence_type == "query_encode":
            token_vecs = token_vecs * token_mask.unsqueeze(-1)
            if self.use_contextualized_stopwords and stopword_importance is not None:
                token_mask[stopword_importance.squeeze(-1) <= 0] = False

            return cls_vec, token_vecs, token_mask
        else:
            return cls_vec, token_vecs, token_mask, stopword_importance

    def forward_shared_encoding(self,  # type: ignore
                                tokens: Dict[str, torch.LongTensor]):
        #
        # ColBERT-style encoding
        #
        mask = tokens["attention_mask"].bool()
        vecs = self.bert_model(input_ids=tokens["input_ids"],
                               attention_mask=tokens["attention_mask"])[0]

        #
        # ColBERTer: Enhanced Reduction
        # ------------------------------------

        #
        # Part 1: Retrieval, Compress [CLS] vectors only to self.compressor_retrieval dim
        #
        if self.use_retrieval_compression:
            retrieval_vec = self.compressor_retrieval(vecs[:, 0, :])
        else:
            retrieval_vec = vecs[:, 0, :]

        #
        # Part 2, Refinement 2.1: Dimensionality Reduction of Compressed Sub-Word Representations
        # -> seems counter-intuitive to do first, but saves a lot of memory consumption for the aggregation
        #
        if self.use_compressor:
            vecs = self.compressor(vecs)

        #
        # Part 2, Refinement 2.2: Aggregate unique-whole-words
        #
        if self.aggregate_unique_ids:

            # aggregate sub-words
            # unique_ids = document["unique_input_ids"] #torch.unique(document["input_ids"],dim=1)
            #aggregation_mask = (unique_ids.unsqueeze(-1) == document["input_ids"].unsqueeze(1)).unsqueeze(-1)

            # aggregate whole-words
            aggregation_mask = (tokens["unique_words"].unsqueeze(-1) == tokens["input_ids_to_words_map"].unsqueeze(1)).unsqueeze(-1)
            aggregated_vecs = (vecs.unsqueeze(1).expand(-1, aggregation_mask.shape[1], -1, -1)*aggregation_mask).sum(2)

            # mean pooling (instead of sum)
            aggregated_vecs = aggregated_vecs / aggregation_mask.float().sum(-2)

            mask = tokens["unique_words"] > 0
            vecs = aggregated_vecs

        return retrieval_vec, vecs, mask

    def forward_aggregation(self, cls_score, query_vecs, query_mask, document_vecs, document_mask,return_all_scores=False, exact_scoring_mask=None):

        score_per_term = torch.bmm(query_vecs, document_vecs.transpose(2, 1))

        if exact_scoring_mask is not None:
            score_per_term[~exact_scoring_mask] = 0

        score_per_term[~(document_mask).unsqueeze(1).expand(-1, score_per_term.shape[1], -1)] = - 1000
        score_refine = score_per_term.max(-1).values
        score_refine[~query_mask] = 0
        score_refine = score_refine.sum(-1)

        #
        # ColBERTer: Add together the two score options
        #
        weight = torch.sigmoid(self.score_merger)
        score = (weight * cls_score) + ((1 - weight) * score_refine)

        if return_all_scores:
            return score, score_refine, score_per_term
        else:
            return score

    def get_param_stats(self):
        return "ColBERTer: "

    def get_param_secondary(self):
        return {}

    def get_output_dim(self):
        dim_info = namedtuple('dim_info', ['cls_vector', "token_vector"])

        if hasattr(self.bert_model.config, "dim"):
            cls_vec_dim = self.bert_model.config.dim
            token_vec_dim = self.bert_model.config.dim
        else:
            cls_vec_dim = self.bert_model.config.hidden_size
            token_vec_dim = self.bert_model.config.hidden_size

        if self.use_second_compression:
            token_vec_dim = self.mini_compressor.out_features
        elif self.use_compressor:
            token_vec_dim = self.compressor.out_features
            
        if self.use_retrieval_compression:
            cls_vec_dim = self.compressor_retrieval.out_features

        return dim_info(cls_vector=cls_vec_dim, token_vector=token_vec_dim)
