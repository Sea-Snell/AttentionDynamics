import torch
import json
import math
import random
import numpy as np
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from seq2seq_model import Seq2seq
from argparse import ArgumentParser
import numpy as np
import pickle as pkl
import os
from typing import List
from load_datasets import DataManager

epsilon = 1e-6

def op_on_hidden(op1, op2, hc):
	h1, c1 = hc[0]
	if hc[1] is not None:
		h2, c2 = hc[1]
		return (op1(h1), op1(c1)), (op2(h2), op2(c2))
	return (op1(h1), op1(c1)), (None, None)



def predict_beam(model, sentences, pad_id, k=5, max_length=100):
	"""Make predictions for the given inputs using beam search.

	Args:
		model: A sequence-to-sequence model.
		sentences: A list of input sentences, represented as strings.
		k: The size of the beam.
		max_length: The maximum length at which to truncate outputs in order to
			avoid non-terminating inference.

	Returns:
		A list of beam predictions. Each element in the list should be a list of k
		strings corresponding to the top k predictions for the corresponding input,
		sorted in descending order by score.
	"""

	# Requirement: your implementation must be batched. This means that you should
	# make only one call to model.encode() at the start of the function, and make
	# only one call to model.decode() per inference step.

	# Suggestion: for efficiency, we suggest that you implement all beam
	# manipulations using batched PyTorch computations rather than Python
	# for-loops.

	# Implementation tip: once an EOS token has been generated, force the output
	# for that candidate to be padding tokens in all subsequent time steps by
	# adding a large positive number like 1e9 to the appropriate logits. This
	# will ensure that the candidate stays on the beam, as its probability
	# will be very close to 1 and its score will effectively remain the same as
	# when it was first completed.  All other (invalid) token continuations will
	# have extremely low log probability and will not make it onto the beam.

	# Implementation tip: while you are encouraged to keep your tensor dimensions
	# constant for simplicity (aside from the sequence length), some special care
	# will need to be taken on the first iteration to ensure that your beam
	# doesn't fill up with k identical copies of the same candidate.

	# YOUR CODE HERE
	# prepare all the result aggregators
	batch_size = len(sentences)
	hypothesis_count = batch_size * k
	hypothesis_ends = torch.zeros(hypothesis_count, dtype=torch.bool).to(device)
	hypothesis_scores = torch.zeros(hypothesis_count, dtype=torch.float).to(device)
	all_hypothesis = torch.tensor([[bos_id for _ in range(hypothesis_count)]]).to(device)
	source = make_batch(sentences, additional_eos=False)
	encoder_output, encoder_mask, decoder_hidden = model.encode(source)
	hidden_dim = encoder_output.shape[-1] // 2

	# take care of the first step of beam search
	next_tokens = torch.tensor([[bos_id for _ in range(batch_size)]]).to(device)

	logits, decoder_hidden, attention_weights = model.decode(
					next_tokens, decoder_hidden, encoder_output, encoder_mask)
	logits = logits.squeeze(0)
	topk_value, topk_idx = torch.topk(logits, k=k, axis=-1)
	flattend_toks = topk_idx.reshape(1, -1)
	next_tokens = flattend_toks
	hypothesis_scores += topk_value.reshape(-1)
	hypothesis_ends = hypothesis_ends | (flattend_toks.reshape(-1) == eos_id)
	all_hypothesis = torch.cat((all_hypothesis, next_tokens), axis=0)

	# repeat the tensor k times
	repeat_indicator = torch.arange(0, batch_size).unsqueeze(0).repeat(k, 1).transpose(0, 1).flatten()
	encoder_output = encoder_output[:, repeat_indicator, :]
	encoder_mask = encoder_mask[:, repeat_indicator]
	decoder_hidden = op_on_hidden(lambda h: h[:, repeat_indicator, :], lambda h: h[:, repeat_indicator, :], decoder_hidden)
	pad_lsm = torch.ones(VOCAB_SIZE).to(device) * (-1e9)
	pad_lsm[pad_id] = 0

	for step in range(max_length - 2):
		logits, decoder_hidden, attention_weights = model.decode(
					next_tokens, decoder_hidden, encoder_output, encoder_mask)
		logits = logits.squeeze(0)
		# mask out the non_pad token after the eos
		logits[hypothesis_ends] = pad_lsm

		# get the top k token for each logits
		topk_value, topk_idx = torch.topk(logits, k=k, axis=-1)
		# repeating hypothesis scores
		repeated_hypothesis_scores = hypothesis_scores.repeat(k, 1).transpose(0, 1).reshape(batch_size, k * k)

		# get the logit score for top k word for each top k hypothesis
		topk_idx = topk_idx.squeeze(0).reshape(batch_size, k * k)

		# get the top k value for each top k tokens for each hypothesis
		topk_value = topk_value.squeeze(0).reshape(batch_size, k * k)
		extended_hypothesis_scores = repeated_hypothesis_scores + topk_value

		# get the top scores for all hypothesis
		topk_hypothesis_values, topk_hypothesis_idx = torch.topk(extended_hypothesis_scores, k=k, dim=-1)

		# get the index of the old hypothesis
		topk_old_hypothesis_idx = topk_hypothesis_idx // k

		next_tokens = torch.gather(topk_idx, 1, topk_hypothesis_idx).reshape(1, -1)
		hypothesis_ends = ((next_tokens == pad_id) | (next_tokens == eos_id)).flatten()

		all_hypothesis = all_hypothesis.reshape(step + 2, batch_size, k)
		hypothesis_gather_idx = topk_old_hypothesis_idx.unsqueeze(0).repeat(step + 2, 1, 1)
		all_hypothesis = all_hypothesis.gather(2, hypothesis_gather_idx)
		all_hypothesis = torch.cat((all_hypothesis.reshape(step + 2, -1), next_tokens), dim=0)

		def filter_hidden(h, num_layers, h_dim):
			h = h.reshape(num_layers, batch_size, k, h_dim).permute((1, 2, 0, 3))
			gather_idx = topk_old_hypothesis_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_layers, h_dim)
			selected_h = h.gather(1, gather_idx)
			h = selected_h.permute(2, 0, 1, 3).reshape(num_layers, k * batch_size, h_dim)
			return h.contiguous()
		# get the new hidden states
		decoder_hidden = op_on_hidden(lambda x: filter_hidden(x, NUM_LAYERS, hidden_dim), lambda x: filter_hidden(x, 1, 2 * hidden_dim), decoder_hidden)
		hypothesis_scores = topk_hypothesis_values.reshape(-1)

		if hypothesis_ends.all():
			break
	all_hypothesis = all_hypothesis.T.reshape(batch_size, k, -1)
	all_sents = [[vocab.DecodeIds(sent.cpu().numpy().tolist()) for sent in k_sents] for k_sents in all_hypothesis]
	return all_sents



def evaluate_next_token(model, data_manager, batch_size=64):
	"""Compute token-level perplexity and accuracy metrics.

	Note that the perplexity here is over subwords, not words.

	This function is used for validation set evaluation at the end of each epoch
	and should not be modified.
	"""
	model.eval()
	total_cross_entropy = 0.0
	total_predictions = 0
	correct_predictions = 0
	val_attentions = []
	with torch.no_grad():
		for source, target in data_manager.make_batch_iterator(batch_size):
			encoder_output, encoder_mask, encoder_hidden = model.encode(source)
			decoder_input, decoder_target = target[:-1], target[1:]
			logits, decoder_hidden, attention_weights = model.decode(
					decoder_input, encoder_hidden, encoder_output, encoder_mask)
			for idx, attention_matrix in enumerate(attention_weights.permute((1, 0,  2))):
				src_length = torch.sum(source[:, idx] != data_manager.pad_id).item()
				tgt_length = torch.sum(decoder_target[:, idx] != data_manager.pad_id).item()
				val_attentions.append(attention_matrix[:tgt_length, :src_length].cpu().numpy())
			total_cross_entropy += F.cross_entropy(
					logits.permute(1, 2, 0), decoder_target.permute(1, 0),
					ignore_index=data_manager.pad_id, reduction="sum").item()
			total_predictions += (decoder_target != data_manager.pad_id).sum().item()
			# print(logits.argmax(2), decoder_target)
			correct_predictions += (
					(decoder_target != data_manager.pad_id) &
					(decoder_target == logits.argmax(2))).sum().item()
	perplexity = math.exp(total_cross_entropy / total_predictions)
	accuracy = 100 * correct_predictions / total_predictions
	for val_attn in val_attentions:
		for distr in val_attn:
			assert np.abs(np.sum(distr) - 1) < epsilon

	return perplexity, accuracy, val_attentions


def predict_greedy(model, sentences, pad_id, max_length=100):
	"""Make predictions for the given inputs using greedy inference.

	Args:
		model: A sequence-to-sequence model.
		sentences: A list of input sentences, represented as strings.
		max_length: The maximum length at which to truncate outputs in order to
			avoid non-terminating inference.

	Returns:
		A list of predicted translations, represented as strings.
	"""

	# Requirement: your implementation must be batched. This means that you should
	# make only one call to model.encode() at the start of the function, and make
	# only one call to model.decode() per inference step.

	# Implementation tip: once an EOS token has been generated, force the output
	# for that example to be padding tokens in all subsequent time steps by
	# adding a large positive number like 1e9 to the appropriate logits.

	# YOUR CODE HERE
	source = make_batch(sentences, additional_eos=True)
	batch_size = len(sentences)
	encoder_output, encoder_mask, encoder_hidden = model.encode(source)

	# initialize the beam search
	batch_hypothesis = torch.tensor([[bos_id] for _ in range(batch_size)]).to(device)
	hypothesis_ends = torch.zeros(batch_size, dtype=torch.bool).to(device)
	next_words = [[bos_id for _ in range(batch_size)]]
	decoder_hidden = encoder_hidden
	for time_step in range(max_length):
		decoder_input = torch.tensor(next_words).to(device)
		logits, decoder_hidden, attention_weights = model.decode(
					decoder_input, decoder_hidden, encoder_output, encoder_mask)
		logits = logits.squeeze(0)
		logits[:, pad_id] += hypothesis_ends * 1e9
		best_toks = torch.argmax(logits, dim=-1)
		next_words = best_toks.unsqueeze(0)
		batch_hypothesis = torch.cat((batch_hypothesis, best_toks.unsqueeze(1)), dim=-1)
		hypothesis_ends = hypothesis_ends | (best_toks == eos_id)
	return [vocab.DecodeIds(hypothesis.cpu().numpy().tolist()) for hypothesis in batch_hypothesis]


def evaluate(model, data_manager, batch_size=64, method="greedy"):
	assert method in {"greedy", "beam"}
	source_sentences = [example.src for example in data_manager.dataset]
	target_sentences = [example.trg for example in data_manager.dataset]
	model.eval()
	predictions = []
	with torch.no_grad():
		for start_index in range(0, len(source_sentences), batch_size):
			if method == "greedy":
				prediction_batch = predict_greedy(
						model, source_sentences[start_index:start_index + batch_size], data_manager.pad_id)
			else:
				prediction_batch = predict_beam(
						model, source_sentences[start_index:start_index + batch_size], data_manager.pad_id)
				prediction_batch = [candidates[0] for candidates in prediction_batch]
			predictions.extend(prediction_batch)
	return sacrebleu.corpus_bleu(predictions, [target_sentences]).score


