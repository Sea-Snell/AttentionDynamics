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

def train(model, num_epochs, batch_size, model_file, ref_attn_func=None, attn_only=False, custom_saves=set()):
	"""Train the model and save its best checkpoint.

	Model performance across epochs is evaluated using token-level accuracy on the
	validation set. The best checkpoint obtained during training will be stored on
	disk and loaded back into the model at the end of training.
	"""
	if attn_only:
		optimizer = torch.optim.Adam(model.attn_param_group)
	else:
		optimizer = torch.optim.Adam(model.parameters())
	best_accuracy = 0.0
	evaluate_next_token(model, val_data_manager)
	step_idx = 0

	for epoch in range(num_epochs):
		batch_iterator = make_batch_iterator(train_data_manager, batch_size, shuffle=True)
		print('epoch %d' % epoch)
		model.train()
		for i, (source, target) in enumerate(batch_iterator, start=1):

			optimizer.zero_grad()
			loss_dict = model.compute_loss(source, target)
			loss = loss_dict['loss']
			clf_loss, attn_loss = loss_dict['clf'], loss_dict['attn']

			if step_idx % save_every == 0 or step_idx in custom_saves:
				print('step: %d, loss: %f' % (step_idx, clf_loss))
				torch.save(model.state_dict(), model_file + str(step_idx))

			loss.backward()
			optimizer.step()
			step_idx += 1
		validation_perplexity, validation_accuracy, validation_attention = evaluate_next_token(model, val_data_manager)
		print('validation accuracy: %f' % (validation_accuracy))
		if validation_accuracy > best_accuracy:
			print(
					"Obtained a new best validation accuracy of {:.2f}, saving model "
					"checkpoint to {}...".format(validation_accuracy, model_file))
			torch.save(model.state_dict(), model_file)
			best_accuracy = validation_accuracy

def get_args():
	parser = ArgumentParser()
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--uniform', default=False, action='store_true')
	parser.add_argument('--seed', type=int)
	parser.add_argument('--save_every', type=int, default=2000)
	parser.add_argument('--config', type=str, default='configs/model.json')
	parser.add_argument('--custom_saves', type=str, default=None, help='comma seperated list of iterations to checkpoint')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = get_args()

	assert torch.cuda.is_available()
	# device = torch.device("cpu")
	device = torch.device("cuda")
	print("Using device:", device)

	dataset = args.dataset
	training_data, validation_data, vocab = load_dataset_by_name(dataset)

	pad_id = vocab.PieceToId("<pad>")
	bos_id = vocab.PieceToId("<s>")
	eos_id = vocab.PieceToId("</s>")

	with open(args.config, 'r') as f:
		model_config = json.load(f)

	NUM_LAYERS = model_config['num_layers']
	DROPOUT = model_config['dropout']
	HIDDEN_DIM = model_config['hidden_dim']
	batch_size = model_config['batch_size']

	val_data_manager = StateManager(validation_data, vocab, bos_id, eos_id, pad_id, device, model_config)
	train_data_manager = StateManager(training_data, vocab, bos_id, eos_id, pad_id, device, model_config)
	VOCAB_SIZE = vocab.GetPieceSize()

	save_every = args.save_every
	num_epochs = args.epochs
	seed = args.seed
	if args.custom_saves is None:
		custom_saves = set()
	else:
		custom_saves = set(map(int, args.custom_saves.split(',')))

	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)

	if args.uniform:
		config_name = 'h_dim=%d,dropout=%f,b_size=%d,seed=%d,uniform' % (HIDDEN_DIM, DROPOUT, batch_size, seed)
	else:
		config_name = 'h_dim=%d,dropout=%f,b_size=%d,seed=%d,normal' % (HIDDEN_DIM, DROPOUT, batch_size, seed)
	
	model_path = ("models/%s/%s/" % (dataset, config_name))
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	model = Seq2seq(device=device, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT,
									attn_lambda=0.0, pad_id=pad_id, full_model=True, invasive_uniform=args.uniform).to(device)
	train(model, num_epochs, batch_size, os.path.join(model_path, 'model'), custom_saves=custom_saves)

	model.load_state_dict(torch.load(os.path.join(model_path, 'model')))
	print("BLEU score with beam search ", evaluate(model, val_data_manager, method='beam'))


