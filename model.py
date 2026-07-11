# TODO implement radix tree ds here, 
# for now it is biased to bigram or fuck it 
# refac this class 
import random  
import pickle
from typing import Self 
from pathlib import Path 

from tokenizer import BaseTokenizer
from preproc import END_SEQ, START_SEQ

class Model: 
    def __init__(
        self, 
        ngram: int | None, 
        tokenizer: BaseTokenizer | None, 
        export_count_path: Path | None = Path('out_count.pkl'), 
        export_prob_path: Path | None = Path('out_prob.pkl'), 
    ): 
        self.ngram = ngram 
        if self.ngram and not self.ngram > 1: 
            raise ValueError('at least bigram should be provided (ngram=2)')
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
            token = self.tokens[i]
            if token.strip() == END_SEQ or token == '\n': 
                continue 
            
            # form key, which is ngram tuple 
            key_ = [] 
            for j in range(0, self.ngram):                 
                if (i+j) > len(self.tokens) - 1: break  # out of range  
                if self.tokens[i+j] == '\n': break  # ngram moves to next text 
                token = self.tokens[i+j]
                key_.append(token)
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

    def sample(self, tokens: list[str] = None, stream: bool = False) -> list[str]: 
        """unifrmly sample until END_SEQ token"""
        self.empty_prob_dict_check()
        ngram = len(next(iter(self.prob_dict))) # next(iter(...)) to not cnvrt keys view into list 

        if tokens is None:   
            if ngram == 2:
                tokens = [START_SEQ]
            else: 
                tokens = list(self.sample_start_token()[:-1])
        else: 
            assert len(tokens) == ngram-1 

        keys_ = self.prob_dict.keys()
        sampled = tokens
        while tokens[-1].strip() != END_SEQ: 
            ngrams = [key_ for key_ in keys_ if ''.join(key_[:ngram-1]) == ''.join(tokens)]
            if len(ngrams) == 0:  # token was not found in prob_dict
                return []
            probs = [self.prob_dict[ngram] for ngram in ngrams]
            # uniformly sample here from distr 
            sample = random.choices(population=ngrams, weights=probs, k=1)[0]
            sampled_token = sample[-1]
            sampled.append(sampled_token)
            
            tokens = sample[1:]

            if stream: 
                print(sampled_token, flush=True, end="")

        return sampled
    
    def sample_start_token(self):
        keys_ = self.prob_dict.keys()
        start_tokens = [key_ for key_ in keys_ if key_[0] == START_SEQ]
        return random.choice(start_tokens)

    @classmethod
    def from_pretrained(
        cls, 
        export_count_path: Path | None, 
        export_prob_path: Path | None, 
    ) -> Self: 
        model = cls(
            tokenizer=None, 
            ngram=None, 
            export_count_path=export_count_path,
            export_prob_path=export_prob_path
        ) 
        
        if export_count_path: model.load_count_dict()
        if export_prob_path: model.load_prob_dict() 
        
        return model 