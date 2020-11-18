import pickle as pkl
from argparse import ArgumentParser
from torch import nn
import torch
from tqdm import tqdm
import random
import os
from load_datasets import process_dataset, DataManager
from collections import defaultdict

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--bsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--test_set_size', type=int, default=4000)
    args = parser.parse_args()
    return args

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, emd_dim, pad_id):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=emd_dim, padding_idx=pad_id)
        self.output_linear = nn.Linear(emd_dim, 1)
        self.lsm = nn.BCELoss()
        self.embed_dim = embed_dim
        self.pad_id = pad_id

    def forward(self, src):
        try:
            batched = len(src.shape) != 1
            if not batched:
                src = src.unsqueeze(dim=0)
        except:
            batched = True
            lengths = [len(item) for item in src]
            pad_src = []
            for item in src:
                pad_src.append(item + [self.pad_id] * (max(lengths) - len(item)))
            src = torch.tensor(pad_src).to(device)
        emb = torch.sum(self.embedding(src), dim=1)
        lengths = torch.sum(src != self.pad_id, dim=1).unsqueeze(-1).repeat((1, self.embed_dim))
        emb = emb / lengths
        output_sigmoid = torch.sigmoid(self.output_linear(emb))
        if not batched:
            output_sigmoid = output_sigmoid[0]
        return output_sigmoid

    def obtain_beta(self):
        embedding_matrix = self.embedding.weight.detach().cpu()
        linear_matrix = self.output_linear.weight.detach().cpu()
        beta = embedding_matrix.matmul(linear_matrix.T).squeeze().detach().cpu().numpy()
        return beta

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = get_args()
    dataset = args.dataset
    bsize = args.bsize
    embed_dim = args.embed_dim
    num_epochs = args.epochs
    test_set_size = args.test_set_size

    train_data, val_data = process_dataset(dataset, test_set_size)
    best_loss = float('inf')
    model = EmbeddingModel(vocab_size=train_data.vocab_size, emd_dim=embed_dim, pad_id=train_data.stoi['<pad>']).to(device)

    model.eval()
    total_loss = 0
    for i in tqdm(range(len(val_data.X))):
        src = val_data.X[i]
        trg = val_data.Y[i]
        src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device)
        output_sigmoid = model(src)
        total_loss += -torch.sum(trg * torch.log(output_sigmoid) + (1 - trg) * torch.log(1 - output_sigmoid)).item()
    print(total_loss)

    # optim = torch.optim.SGD(lr=5e-3, params=model.parameters())
    optim = torch.optim.Adam(params=model.parameters(), lr=0.0003)
    best_beta = model.obtain_beta()
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        idxs = list(range(len(train_data.X)))
        random.shuffle(idxs)
        train_data.X, train_data.Y = [train_data.X[idx] for idx in idxs], [train_data.Y[idx] for idx in idxs]

        for i in tqdm(range(0, len(train_data.X), bsize)):
            src_batch, trg = train_data.X[i:(i+bsize)], train_data.Y[i:(i+bsize)]
            output_sigmoid = model(src_batch)
            loss = 0.
            for i in range(len(src_batch)):
                loss -= torch.sum(trg[i] * torch.log(output_sigmoid[i]) + (1 - trg[i]) * torch.log(1 - output_sigmoid[i]))
            loss.backward()
            optim.step()
            optim.zero_grad()

        model.eval()
        total_loss = 0
        for i in tqdm(range(len(val_data.X))):
            src = val_data.X[i]
            trg = val_data.Y[i]
            src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device)
            output_sigmoid = model(src)
            total_loss += -torch.sum(trg * torch.log(output_sigmoid) + (1 - trg) * torch.log(1 - output_sigmoid)).item()
        if total_loss < best_loss:
            best_loss = total_loss
            best_beta = model.obtain_beta()
            best_epoch = epoch
        print('epoch %d, loss %.3f' % (epoch, total_loss))

    if not os.path.exists('outputs/'):
        os.makedirs('outputs/')
    f_path = os.path.join('outputs', '{dataset_name}embedding{embed_dim}beta.pkl'.format(dataset_name=dataset, embed_dim=embed_dim))
    pkl.dump(best_beta, open(f_path, 'wb'))
    print('best loss %.3f from epoch %d saved' % (best_loss, best_epoch))



