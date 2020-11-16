import pickle as pkl
from argparse import ArgumentParser
from torch import nn
import torch
from tqdm import tqdm
import random
from load_datasets import load_dataset_by_name, StateManager, make_batch, make_batch_iterator, sentence2ids_nopad
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--bsize', type=int, default=64)
args = parser.parse_args()

num_epochs = args.epochs
embed_dim = args.embed_dim
bsize = args.bsize
dataset_name = args.dataset

training_data, validation_data, vocab = load_dataset_by_name(dataset_name)
pad_id = vocab.PieceToId("<pad>")
bos_id = vocab.PieceToId("<s>")
eos_id = vocab.PieceToId("</s>")
val_data_manager = StateManager(validation_data, vocab, bos_id, eos_id, pad_id, device, {})
train_data_manager = StateManager(training_data, vocab, bos_id, eos_id, pad_id, device, {})

# train_set = [d for d in dataset if d['split'] == 'train']
# val_set = [d for d in dataset if d['split'] == 'val']
vocab_size = len(vocab)
output_size = len(vocab)
print(vocab_size, output_size)

class EmbeddingModel(nn.Module):

    def __init__(self, vocab_size, output_size, emd_dim, pad_id):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=emd_dim, padding_idx=pad_id)
        self.output_linear = nn.Linear(emd_dim, output_size)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.embed_dim = embed_dim
        self.pad_id = pad_id

    def forward(self, src):
        batched = len(src.shape) != 1
        if not batched:
            src = src.unsqueeze(dim=0)
        emb = torch.sum(self.embedding(src), dim=1)
        lengths = torch.sum(src != pad_id, dim=1).unsqueeze(-1).repeat((1, self.embed_dim))
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

best_loss = float('inf')
model = EmbeddingModel(vocab_size=vocab_size, output_size=output_size, emd_dim=embed_dim, pad_id=pad_id).to(device)

model.eval()
total_loss = 0
for d in tqdm(val_data_manager.dataset):
    src = sentence2ids_nopad(val_data_manager, d.src, additional_eos=False)
    trg = sentence2ids_nopad(val_data_manager, d.trg, additional_eos=False)[1:]
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
    random.shuffle(train_data_manager.dataset)

    for src_batch, trg_batch in tqdm(list(make_batch_iterator(train_data_manager, bsize))):
        src_batch, trg_batch = src_batch.transpose(0, 1), trg_batch.transpose(0, 1)
        src_batch = torch.tensor(src_batch).to(device)
        output_lsm = model(src_batch)
        loss = 0.
        for i in range(len(src_batch)):
            loss -= torch.sum(output_lsm[i][trg_batch[i]])
        loss.backward()
        optim.step()
        optim.zero_grad()

    model.eval()
    total_loss = 0
    for d in tqdm(val_data_manager.dataset):
        src = sentence2ids_nopad(val_data_manager, d.src, additional_eos=False)
        trg = sentence2ids_nopad(val_data_manager, d.trg, additional_eos=False)[1:]
        src, trg = torch.tensor(src).to(device), torch.tensor(trg).to(device)
        output_lsm = model(src)
        total_loss += -torch.sum(output_lsm[trg]).item()
    if total_loss < best_loss:
        best_loss = total_loss
        best_word_translation = model.obtain_word_translation()
        best_epoch = epoch
    print('epoch %d, loss %.3f' % (epoch, total_loss))

if not os.path.exists('outputs/'):
    os.makedirs('outputs/')
f_path = os.path.join('outputs', '{dataset_name}embedding{embed_dim}translation.pkl'.format(dataset_name=dataset_name, embed_dim=embed_dim))
pkl.dump(best_word_translation, open(f_path, 'wb'))
print('best loss %.3f from epoch %d saved' % (best_loss, best_epoch))

