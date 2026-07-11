import argparse

from model import Model
from tokenizer import BaseTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample from pretrained ngram model')
    parser.add_argument("prob_path", type=str)
    args = parser.parse_args() 

    model = Model.from_pretrained(export_count_path=None, export_prob_path=args.prob_path)

    # sampling  
    token = None
    while True: 
        if token:
            token = ' ' + token.strip().lower()
        sampled = model.sample(token) 
        print(''.join(sampled))
        token = input('Enter: ')