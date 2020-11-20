import torch
import torchtext

import json
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
import sentencepiece
import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq_model import Seq2seq
from argparse import ArgumentParser
import numpy as np
import pickle as pkl
import os
from typing import List
from load_datasets import load_dataset_by_name, StateManager, make_batch, make_batch_iterator
from eval_model import evaluate, evaluate_next_token

# class Item:
# 	def __init__(self, src, trg):
# 		self.src = src
# 		self.trg = trg

class DataGen:
	def __init__(self, n, seq_len):
		self.n = n
		self.seq_len = seq_len
		self.vocab_size = n + 3
		self.bos = n
		self.eos = n + 1
		self.pad = n + 2

	def gen_permutation(self):
		return np.array([self.bos] + np.random.choice(self.n, size=self.seq_len).tolist() + [self.eos])

	def batch_permutation(self, b_size):
		return np.stack([self.gen_permutation() for _ in range(b_size)], axis=0).transpose()

# def copy_data_manager(dataset_size, n, seq_len, device, config):
# 	seqs = batch_permutation(dataset_size, n, seq_len)
# 	eos = n
# 	bos = n + 1
# 	pad = n + 2
# 	items = [Item([bos] + seq + [eos], [bos] + seq + [eos]) for seq in seqs]
# 	class TempVocab:
# 		def GetPieceSize(self):
# 			return n
# 	vocab = TempVocab()
# 	return StateManager(items, vocab, bos, eos, pad, device, config)

def attention_accuracy(attention, grad=-1):
	max_ = torch.argmax(attention, dim=-1).transpose(0, 1)
	total = 0.0
	count = 0
	if grad == 2:
		print(max_[:, :-1])
	for item in max_:
		count += len(item)
		total += torch.sum(item[:-1] == torch.tensor(list(range(1, len(item)))))
	return total / count

def get_grad_influence2(model, data_gen, batches, batch_size):
	assert model.dropout == 0
	model.train()
	influences = []
	for _ in range(batches):
		source = target = data_gen.batch_permutation(batch_size)
		source = torch.tensor(source)
		target = torch.tensor(target)
		encoder_output, encoder_mask, encoder_hidden = model.encode(source, in_grad=True)
		decoder_input, decoder_target = target[:-1], target[1:]
		logits, decoder_hidden, attention_weights = model.decode(
		    decoder_input, encoder_hidden, encoder_output, encoder_mask)
		# print('acc 2 logits', logits.shape)
		# print('acc 2 input', source.shape, model.inputs.shape)
		grads = []
		for i in range(decoder_target.shape[0]):
			grad = model.influence(-logits[i, list(range(len(decoder_target[i]))), decoder_target[i]], retain_graph=(i != decoder_target.shape[0] - 1))
			grads.append(grad)
		grads = torch.stack(grads, dim=0).permute(2, 0, 1)
		for idx in range(len(grads)):
			tgt_length = torch.sum(decoder_target[:, idx].squeeze() != data_gen.pad).item()
			src_length = torch.sum(source[:, idx].squeeze() != data_gen.pad).item()
			influences.append((-grads[idx][:tgt_length, :src_length]).clone().detach().cpu().numpy())
	return influences

def get_grad_influence3(model, data_gen, batches):
	assert model.dropout == 0
	model.train()
	influences = []
	for batch in range(batches):
		source = target = data_gen.batch_permutation(1)
		source = torch.tensor(source)
		target = torch.tensor(target)
		src_emb = model.embedding(source).squeeze(1)
		encoder_output, encoder_mask, encoder_hidden = model.encode(source)
		decoder_input, decoder_target = target[:-1], target[1:]
		logits, decoder_hidden, attention_weights = model.decode(
		    decoder_input, encoder_hidden, encoder_output, encoder_mask)
		logits = logits.squeeze()
		decoder_target = decoder_target.squeeze()
		tgt_length = torch.sum(decoder_target.squeeze() != data_gen.pad).item()

		model.embedding(source).squeeze(1)

		grads = []
		for i in range(tgt_length):
			# negative log loss
			model.zero_grad()
			(-1 * logits[i][decoder_target[i]]).backward(retain_graph=(i != tgt_length - 1))
			grad = model.grads['source_emb'].clone().detach().squeeze(1)
			grads.append(grad.unsqueeze(0))
		grads = torch.cat(grads, dim=0)
		# increase in the grad direction will increase the loss
		# dot product with grad means: the loss increase if the src_embedding goes from zero to the current position
		# therefore, the less of src_emb dot grad, the more the occurrence of this source token contributes to the occurence of the target
		# a positive score in the influence tensor indicates a positive influence
		influence = -torch.einsum("tsh,sh->ts", [grads, src_emb]).detach().cpu().numpy()
		influences.append(influence)
	model.zero_grad()
	return influences

def train(model, num_steps, batch_size, data_gen):
	"""Train the model and save its best checkpoint.

	Model performance across epochs is evaluated using token-level accuracy on the
	validation set. The best checkpoint obtained during training will be stored on
	disk and loaded back into the model at the end of training.
	"""
	optimizer = torch.optim.Adam(model.parameters())
	step_idx = 0
	model.train()

	for step_idx in range(num_steps):
		source = target = data_gen.batch_permutation(batch_size)
		optimizer.zero_grad()
		loss_dict = model.compute_loss(torch.tensor(source).to(device), torch.tensor(target).to(device))
		loss = loss_dict['loss']
		clf_loss, attn_loss = loss_dict['clf'], loss_dict['attn']

		if step_idx % 10 == 0:
			print('step: %d, loss: %f' % (step_idx, clf_loss))
			print('attention acc:', attention_accuracy(loss_dict['attention']).item())
			temp_dropout = model.dropout
			model.dropout = 0.0
			print('grad acc 2:', attention_accuracy(torch.stack(list(map(lambda x: torch.tensor(x), get_grad_influence2(model, data_gen, 1, 8))), dim=0).permute(1, 0, 2), grad=2).item())
			print('grad acc 3:', attention_accuracy(torch.stack(list(map(lambda x: torch.tensor(x), get_grad_influence3(model, data_gen, 8))), dim=0).permute(1, 0, 2), grad=3).item())
			model.dropout = temp_dropout
		loss.backward()
		optimizer.step()

if __name__ == '__main__':
	device = torch.device("cpu")
	print("Using device:", device)

	config = './configs/model.json'
	with open(config, 'r') as f:
		model_config = json.load(f)

	NUM_LAYERS = model_config['num_layers']
	DROPOUT = model_config['dropout']
	HIDDEN_DIM = model_config['hidden_dim']
	batch_size = 8
	n = 40
	num_steps = 1000
	seq_len = 10
	data_gen = DataGen(n, seq_len)


	# torch.manual_seed(seed)
	# random.seed(seed)
	# np.random.seed(seed)

	model = Seq2seq(device=device, hidden_dim=HIDDEN_DIM, vocab_size=data_gen.vocab_size, num_layers=NUM_LAYERS, dropout=DROPOUT,
									attn_lambda=0.0, pad_id=data_gen.pad, full_model=True, invasive_uniform=False).to(device)
	train(model, num_steps, batch_size, data_gen)




