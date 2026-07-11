# TODO implement radix tree ds here, 
# for now it is biased to bigram or fuck it 
# refac this class 
import random  
import pickle
from pathlib import Path 

from tokenizer import BaseTokenizer
from preproc import START_SEQ, END_SEQ

class Model: 
    def __init__(
        self, 
        ngram: int | None, 
        tokenizer: BaseTokenizer | None, 
        export_count_path: Path = Path('out_count.pkl'), 
        export_prob_path: Path = Path('out_prob.pkl'), 
    ): 
        self.ngram = ngram 
        if not (2 >= self.ngram >= 1): 
            raise ValueError('Only 1 - unigram and 2 - bigrams are supported')
        self.tokenizer = tokenizer 
        self.export_count_path = export_count_path
        self.export_prob_path = export_prob_path 
        self.count_dict = dict() 
        self.prob_dict = dict()
        self.raw = None
        self.tokens = []

    def load_raw(self, path: Path): 
        with open(path, 'r') as f: 
            self.raw = f.read()
    
    def tokenize(self, lower: bool = True): 
        if not self.raw: 
            raise IOError('no data was loaded, use load_raw() method')

        self.tokens = self.tokenizer.__call__(text=self.raw, lower=lower)

    def count(self): 
        if len(self.tokens) == 0: 
            raise IOError('tokens list is empty, use tokenize() method')

        for i in range(0, len(self.tokens)): 
            token = self.tokens[i].strip()

            if token == END_SEQ or token == '\n': 
                continue 
            
            # form key, which is ngram tuple 
            key_ = [] 
            for j in range(0, self.ngram):
                key_.append(self.tokens[i+j])
            key_ = tuple(key_)

            try:
                count = self.count_dict[key_]
            except KeyError: 
                self.count_dict[key_] = 1 
            else: 
                self.count_dict[key_] = count + 1 
    
    def empty_count_dict_check(self): 
        if len(self.count_dict.keys()) == 0:
            raise IOError('count_dict is empty')

    def empty_prob_dict_check(self): 
        if len(self.prob_dict.keys()) == 0:
            raise IOError('prob_dict is empty') 

    def export_count_dict(self): 
        self.empty_count_dict_check()

        with open(self.export_count_path, 'wb') as f: 
            pickle.dump(self.count_dict, f)

    def load_count_dict(self): 
        with open(self.export_count_path, 'rb') as f: 
            self.count_dict = pickle.load(f)

    def export_prob_dict(self): 
        self.empty_prob_dict_check()

        with open(self.export_prob_path, 'wb') as f: 
            pickle.dump(self.prob_dict, f)

    def load_prob_dict(self): 
        with open(self.export_prob_path, 'rb') as f: 
            self.prob_dict = pickle.load(f)

    def count_prob(self): 
        self.empty_count_dict_check()

        keys_ = self.count_dict.keys()
        
        # count prefixex 
        prefix_counts = dict() 
        for key_ in keys_:
            prefix = key_[:-1]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1 

        # calc prob by dividing count_dict[ngram] / prefix_counts[ngram-1]
        for key_ in keys_: 
            ngram_count = self.count_dict[key_]
            prefix = key_[:-1]
            self.prob_dict[key_] = ngram_count / prefix_counts[prefix]             

    # TODO handle out of range tokens 
    def sample(self, token: str | None) -> list[str]: 
        """unifrmly sample until END_SEQ token, start from START_SEQ token or provided token"""
        self.empty_prob_dict_check()

        token = START_SEQ if not token else token 
        keys_ = self.prob_dict.keys()
        sampled = [token]
        while token.strip() != END_SEQ: 
            ngrams = [key_ for key_ in keys_ if key_[0] == token]
            probs = [self.prob_dict[ngram] for ngram in ngrams]
            # uniformly sample here from distr 
            sample = random.choices(population=ngrams, weights=probs, k=1)[0]
            token = sample[-1]
            sampled.append(token)
        
        return sampled
    