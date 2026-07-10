from model import Model

if __name__ == "__main__":
    model = Model(2, None)
    model.load_prob_dict()
    token = None 
    while True: 
        if token:
            token = ' ' + token.strip().lower()
        sampled = model.sample(token) 
        print(''.join(sampled))
        token = input('Enter: ')