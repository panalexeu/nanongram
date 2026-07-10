from model import Model
from tokenizer import BaseTokenizer

if __name__ == "__main__":
    model = Model(2, BaseTokenizer(), None, None)
    model.load_raw('./outme.txt')
    model.tokenize()
    model.count()
    model.count_prob()
    token = None 
    while True: 
        if token:
            token = ' ' + token.strip().lower()
        sampled = model.sample(token) 
        print(''.join(sampled))
        token = input('Enter: ')