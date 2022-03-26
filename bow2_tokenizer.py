import torch
import numpy
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import PreTrainedTokenizerFast
import hashlib

class BOW2Tokenizer():
    """
    Bag-of-Whole-Words wrapper for a Huggingface Tokenizer.

    We use the BertPreTokenizer to tokenize the input into whole-words (split on punctuation and whitespaces). 
    This has the benefit that whole-word and sub-word boundaries match up exactly, 
    and we don't have to do any weird alignment tricks.

    We augment the data output dictionary of the tokenize() call with the following keys:
    - "unique_ids": a list of unique ids -> need for batching & correct padding later on in the model (if subword token based aggregation is used)
    - "input_ids_to_words_map": a map for each entry in input_ids to the corresponding word in unique_words
    - "unique_words": a list of unique whole-word ids (either a range or a global usable hash if create_global_id is True)

    0 is never used by the additional ids and may be safely used as padding later on

    usage example: 
        tok = BOW2Tokenizer(AutoTokenizer.from_pretrained("distilbert-base-uncased"))
        tok.tokenize("This is a sentence.")
    
    parameters:
        tokenizer: An instantiated Huggingface Tokenizer (BERT-style)
        add_unique_ids: bool, if false returns the huggigface tokenizer output (a no-op for us)
        uniqueness_type: str, how to handle uniqueness of tokens, either "lower" or "stemmed" (we use nltk.stem.porter for stemming)
        create_global_id: bool, whether to create a global hashed-id for each token (the first 32 bits of sha256) or just count up per sentence
    """

    def __init__(self, tokenizer:PreTrainedTokenizerFast, add_unique_ids=True, uniqueness_type="stemed", create_global_id=True):

        self._tokenizer = tokenizer
        self.add_unique_ids = add_unique_ids
        if self.add_unique_ids:
            self.pre_tokenzier = BertPreTokenizer()

            from nltk.stem.porter import PorterStemmer
            self.stemmer = PorterStemmer()
            
            self.uniqueness_type = uniqueness_type
            self.create_global_id = create_global_id

            self.stem_cache = {}

    def tokenize(self, sentence: str, sentence2: str = None, max_length: int = 512):
        if sentence2 != None:
            seq_tokenized = self._tokenizer(sentence, sentence2,
                                            max_length=max_length,
                                            truncation=True,
                                            return_tensors="pt",
                                            return_attention_mask=True)

        else:
            seq_tokenized = self._tokenizer(sentence,
                                            max_length=max_length,
                                            truncation=True,
                                            return_tensors="pt",
                                            return_attention_mask=True)
            #
            # only used for ColBERTer model
            #
            if self.add_unique_ids:

                seq_tokenized.data["unique_input_ids"] = torch.unique(seq_tokenized.data["input_ids"])
                
                # these are the wordpiece-subwords
                tf_offsets = seq_tokenized.encodings[0].offsets

                # these are the whole-word offsets (subwords are not split yet), but it uses the exact same splitting mechanism
                whole_word_offsets = self.pre_tokenzier.pre_tokenize_str(sentence)

                # create unique_token_dict
                whole_word_unique = {}
                for i,(tok,offsets) in enumerate(whole_word_offsets):
                    if self.uniqueness_type == "stemmed":
                        lower_tok = tok.lower()
                        if lower_tok not in self.stem_cache:
                            tok_transformed = self.stemmer.stem(lower_tok)
                            self.stem_cache[lower_tok] = tok_transformed
                        else:
                            tok_transformed = self.stem_cache[lower_tok]
                    else:
                        tok_transformed = tok.lower()

                    whole_word_offsets[i] = (tok_transformed,offsets)
                    
                    if tok_transformed not in whole_word_unique:
                        if self.create_global_id:
                            hashed = int.from_bytes(hashlib.sha256(tok_transformed.encode('utf-8')).digest()[:4], 'little', signed=False) # 32-bit int
                            # 0 is a reserved id for padding, don't think this will happen often though
                            if hashed == 0:
                                hashed = 1
                                
                            if hashed < 0 or hashed > 4294967295:
                                print("Warning: hash value is too large, will be truncated to 32-bit int")
                            whole_word_unique[tok_transformed] = hashed
                        else:
                            whole_word_unique[tok_transformed] = len(whole_word_unique) + 1

                # map tf_offsets to whole_word_unique
                tf_input_ids_to_whole_word_unique_map = torch.zeros_like(seq_tokenized.data["input_ids"])
                for i,tf_offset in enumerate(tf_offsets[1:-1]): # ignore special tokens
                    for whole_word_token,whole_word_offset in whole_word_offsets:
                        if tf_offset[0] >= whole_word_offset[0] and tf_offset[1] <= whole_word_offset[1]:
                            tf_input_ids_to_whole_word_unique_map[0][i+1] = whole_word_unique[whole_word_token]
                            break
                
                # if the tokenizer cuts off the sequence, we might have some tokens that are in the pre-tokenizer, but not mapped
                # because they only appear in the end and where cut -> in this case we just remove them also from the unique list
                # as the main tokenizer is the main anchor point
                skipped_whole_word =[]
                for tok,i in whole_word_unique.items():
                    if i not in tf_input_ids_to_whole_word_unique_map[0]:
                        skipped_whole_word.append(tok)
                for tok in skipped_whole_word:
                    del whole_word_unique[tok]

                #
                # this is just sanity checking to make sure that the mapping is correct
                #
                #if (tf_input_ids_to_whole_word_unique_map[0][1:-1] == 0).any():
                #    missing_ids = seq_tokenized.data["input_ids"][0][1:-1][tf_input_ids_to_whole_word_unique_map[0][1:-1] == 0]
                #    missing_toks = self._tokenizer.convert_ids_to_tokens(missing_ids)
                #    if not (len(set(missing_toks)) == 1 and missing_toks[0] == "[PAD]"):
                #        print("WARNING: some tokens were not found in the whole_word dictionary",missing_toks,"in sentence:", sentence, "with offset:", whole_word_offsets,"unique_words", whole_word_unique)

                seq_tokenized.data["input_ids_to_words_map"] = tf_input_ids_to_whole_word_unique_map
                seq_tokenized.data["unique_words"] = torch.from_numpy(numpy.array(list(whole_word_unique.values()),dtype=numpy.int64)).unsqueeze(0)

        return seq_tokenized
