import pickle as pkl
from torch import nn
import torch
from tqdm import tqdm
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_name = 'multi30k'
dataset = pkl.load(open('data/{dataset_name}_logs_rz.pkl'.format(dataset_name=dataset_name), 'rb'))
train_set = [d for d in dataset if d['split'] == 'train']
val_set = [d for d in dataset if d['split'] == 'val']
vocab_size = 0
for d in train_set:
    vocab_size = max(vocab_size, max(d['src']))
vocab_size += 1
output_size = 0
for d in train_set:
    output_size = max(vocab_size, max(d['trg']))
print(vocab_size, output_size)

class EmbeddingModel(nn.Module):

    def __init__(self, vocab_size, output_size, emd_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=emd_dim, padding_idx=0)
        self.output_linear = nn.Linear(emd_dim, output_size)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.embed_dim = embed_dim

    def forward(self, src):
        batched = len(src.shape) != 1
        if not batched:
            src = src.unsqueeze(dim=0)
        emb = torch.sum(self.embedding(src), dim=1)
        lengths = torch.sum(src != 0, dim=1).unsqueeze(-1).repeat((1, self.embed_dim))
        emb = emb / lengths
        output_lsm = self.lsm(self.output_linear(emb))
        if not batched:
            output_lsm = output_lsm[0]
        return output_lsm

    def obtain_word_translation(self):
        embedding_matrix = self.embedding.weight.detach().cpu()
        linear_matrix = self.output_linear.weight.detach().cpu()
        word_translation = torch.softmax(embedding_matrix.matmul(linear_matrix.T), dim=-1).detach().cpu().numpy()
        return word_translation


num_epochs = 30
embed_dim = 256
bsize = 64
best_loss = float('inf')
model = EmbeddingModel(vocab_size=vocab_size, output_size=output_size, emd_dim=embed_dim).to(device)

model.eval()
total_loss = 0
for d in tqdm(val_set):
    src, trg = d['src'], d['trg'][1:]
    src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device)
    output_lsm = model(src)
    total_loss += -torch.sum(output_lsm[trg]).item()
print(total_loss)

# optim = torch.optim.SGD(lr=5e-3, params=model.parameters())
optim = torch.optim.Adam(params=model.parameters())
best_word_translation = model.obtain_word_translation()
best_epoch = 0
for epoch in range(num_epochs):
    model.train()
    random.shuffle(train_set)

    src_batch, trg_batch = [], []
    for d in tqdm(train_set):
        src, trg = d['src'], d['trg'][1:]
        src_batch.append(src)
        trg_batch.append(trg)
        if len(src_batch) == bsize:
            max_length = max([len(src) for src in src_batch])
            for src in src_batch:
                while len(src) < max_length:
                    src.append(0)
            src_batch = torch.tensor(src_batch).to(device)
            output_lsm = model(src_batch)
            loss = 0.
            for i in range(bsize):
                loss -= torch.sum(output_lsm[i][trg_batch[i]])
            loss.backward()
            optim.step()
            optim.zero_grad()
            src_batch, trg_batch = [], []

    model.eval()
    total_loss = 0
    for d in tqdm(val_set):
        src, trg = d['src'], d['trg'][1:]
        src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device)
        output_lsm = model(src)
        total_loss += -torch.sum(output_lsm[trg]).item()
    if total_loss < best_loss:
        best_loss = total_loss
        best_word_translation = model.obtain_word_translation()
        best_epoch = epoch
    print('epoch %d, loss %.3f' % (epoch, total_loss))


f_path = 'data/{dataset_name}embedding{embed_dim}translation.pkl'.format(dataset_name=dataset_name, embed_dim=embed_dim)
pkl.dump(best_word_translation, open(f_path, 'wb'))
print('best loss %.3f from epoch %d saved' % (best_loss, best_epoch))

