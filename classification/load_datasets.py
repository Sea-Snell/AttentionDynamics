import json
import pickle as pkl
from collections import defaultdict

class DataManager:
    def __init__(self, X, Y, tokenid2vector, vocab_size, stoi):
        self.X = X
        self.Y = Y
        self.tokenid2vector = tokenid2vector
        self.vocab_size = vocab_size
        self.stoi = stoi

def process_dataset(dataset_name, test_set_size):
    train_dat = json.load(open('data/%s_train.json' % (dataset_name)))
    test_dat = json.load(open('data/%s_test.json' % (dataset_name)))
    vocab = train_dat['vocab']

    raw_train_X, train_Y = zip(*(train_dat['XY']))
    textual_token_train_X = [[vocab[token_idx] for token_idx in item] for item in raw_train_X]

    raw_test_X, test_Y = zip(*test_dat['XY'])
    textual_token_test_X = [[vocab[token_idx] for token_idx in item] for item in raw_test_X]
    textual_token_test_X = textual_token_test_X[:test_set_size]
    test_Y = test_Y[:test_set_size]

    w2v = pkl.load(open('data/w2v.pkl', 'rb'))
    itos = ['<pad>', '<bos>', '<eos>', '<unk>']
    vocab_counter = defaultdict(int)
    for seq in textual_token_train_X:
        for token in seq:
            if token not in itos:
                vocab_counter[token] += 1
    sorted_vocab = sorted([t for t in vocab_counter if vocab_counter[t] > 2 or t in w2v], key=lambda x: -vocab_counter[x])
    itos.extend(sorted_vocab)
    stoi = defaultdict(lambda: itos.index('<unk>'))
    for i, v in enumerate(itos):
        stoi[v] = i
    vocab_size = len(itos)
    tokenid2vector = {}
    for i, word in enumerate(itos):
        if word in w2v:
            tokenid2vector[i] = w2v[word]

    token_id_train_X = [[stoi['<bos>']] + [stoi[w] for w in seq] + [stoi['<eos>']] for seq in textual_token_train_X]
    token_id_test_X = [[stoi['<bos>']] + [stoi[w] for w in seq] + [stoi['<eos>']] for seq in textual_token_test_X]
    train_data_manager = DataManager(token_id_train_X, train_Y, tokenid2vector, vocab_size, stoi)
    test_data_manager = DataManager(token_id_test_X, test_Y, tokenid2vector, vocab_size, stoi)
    return train_data_manager, test_data_manager



