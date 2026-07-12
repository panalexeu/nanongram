import argparse

from .model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sample from pretrained ngram model')
    parser.add_argument("prob_path", type=str)
    args = parser.parse_args() 

    model = Model.from_pretrained(export_count_path=None, export_prob_path=args.prob_path)

    # sampling  
    while True: 
        sampled = model.sample(tokens=None, stream=True) 
        input('\nEnter: ')