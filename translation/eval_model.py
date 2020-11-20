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
from load_datasets import make_batch, make_batch_iterator

epsilon = 1e-6

def op_on_hidden(op1, op2, hc):
	h1, c1 = hc[0]
	if hc[1] is not None:
		h2, c2 = hc[1]
		return (op1(h1), op1(c1)), (op2(h2), op2(c2))
	return (op1(h1), op1(c1)), (None, None)



def predict_beam(model, sentences, data_manager, k=5, max_length=100):
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
	hypothesis_ends = torch.zeros(hypothesis_count, dtype=torch.bool).to(data_manager.device)
	hypothesis_scores = torch.zeros(hypothesis_count, dtype=torch.float).to(data_manager.device)
	all_hypothesis = torch.tensor([[data_manager.bos_id for _ in range(hypothesis_count)]]).to(data_manager.device)
	source = make_batch(data_manager, sentences, additional_eos=False)
	encoder_output, encoder_mask, decoder_hidden = model.encode(source)
	hidden_dim = encoder_output.shape[-1] // 2

	# take care of the first step of beam search
	next_tokens = torch.tensor([[data_manager.bos_id for _ in range(batch_size)]]).to(data_manager.device)

	logits, decoder_hidden, attention_weights = model.decode(
					next_tokens, decoder_hidden, encoder_output, encoder_mask)
	logits = logits.squeeze(0)
	topk_value, topk_idx = torch.topk(logits, k=k, axis=-1)
	flattend_toks = topk_idx.reshape(1, -1)
	next_tokens = flattend_toks
	hypothesis_scores += topk_value.reshape(-1)
	hypothesis_ends = hypothesis_ends | (flattend_toks.reshape(-1) == data_manager.eos_id)
	all_hypothesis = torch.cat((all_hypothesis, next_tokens), axis=0)

	# repeat the tensor k times
	repeat_indicator = torch.arange(0, batch_size).unsqueeze(0).repeat(k, 1).transpose(0, 1).flatten()
	encoder_output = encoder_output[:, repeat_indicator, :]
	encoder_mask = encoder_mask[:, repeat_indicator]
	decoder_hidden = op_on_hidden(lambda h: h[:, repeat_indicator, :], lambda h: h[:, repeat_indicator, :], decoder_hidden)
	pad_lsm = torch.ones(data_manager.vocab_size).to(data_manager.device) * (-1e9)
	pad_lsm[data_manager.pad_id] = 0

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
		hypothesis_ends = ((next_tokens == data_manager.pad_id) | (next_tokens == data_manager.eos_id)).flatten()

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
		decoder_hidden = op_on_hidden(lambda x: filter_hidden(x, data_manager.num_layers, hidden_dim), lambda x: filter_hidden(x, 1, 2 * hidden_dim), decoder_hidden)
		hypothesis_scores = topk_hypothesis_values.reshape(-1)

		if hypothesis_ends.all():
			break
	all_hypothesis = all_hypothesis.T.reshape(batch_size, k, -1)
	all_sents = [[data_manager.vocab.DecodeIds(sent.cpu().numpy().tolist()) for sent in k_sents] for k_sents in all_hypothesis]
	return all_sents


def get_state_scores(model, dataset_manager):
    assert model.dropout == 0
    model.train()
    val_data_iter = make_batch_iterator(dataset_manager, 1)
    all_state_scores = []
    for data_id, (source, target) in enumerate(val_data_iter):
        all_state_scores.append(model.get_state_scores(source, target)[0])
    return all_state_scores


def get_state_scores2(model, dataset_manager):
    assert model.dropout == 0
    model.train()
    val_data_iter = make_batch_iterator(dataset_manager, 1)
    all_state_scores = []
    for data_id, (source, target) in enumerate(val_data_iter):
        all_state_scores.append(model.get_state_scores2(source, target)[0])
    return all_state_scores


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
        for source, target in make_batch_iterator(data_manager, batch_size):
            encoder_output, encoder_mask, encoder_hidden = model.encode(source)
            decoder_input, decoder_target = target[:-1], target[1:]
            logits, decoder_hidden, attention_weights = model.decode(decoder_input, encoder_hidden, encoder_output, encoder_mask)
            for idx, attention_matrix in enumerate(attention_weights.permute((1, 0,  2))):
                src_length = torch.sum(source[:, idx] != data_manager.pad_id).item()
                tgt_length = torch.sum(decoder_target[:, idx] != data_manager.pad_id).item()
                val_attentions.append(attention_matrix[:tgt_length, :src_length].cpu().numpy())
            total_cross_entropy += F.cross_entropy(
            logits.permute(1, 2, 0), decoder_target.permute(1, 0), ignore_index=data_manager.pad_id, reduction="sum").item()
            total_predictions += (decoder_target != data_manager.pad_id).sum().item()
            # print(logits.argmax(2), decoder_target)
            correct_predictions += ((decoder_target != data_manager.pad_id) &
					(decoder_target == logits.argmax(2))).sum().item()
    perplexity = math.exp(total_cross_entropy / total_predictions)
    accuracy = 100 * correct_predictions / total_predictions
    for val_attn in val_attentions:
        for distr in val_attn:
            assert np.abs(np.sum(distr) - 1) < epsilon
        
    return perplexity, accuracy, val_attentions


def predict_greedy(model, sentences, data_manager, max_length=100):
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
	source = make_batch(data_manager, sentences, additional_eos=True)
	batch_size = len(sentences)
	encoder_output, encoder_mask, encoder_hidden = model.encode(source)

	# initialize the beam search
	batch_hypothesis = torch.tensor([[data_manager.bos_id] for _ in range(batch_size)]).to(data_manager.device)
	hypothesis_ends = torch.zeros(batch_size, dtype=torch.bool).to(data_manager.device)
	next_words = [[data_manager.bos_id for _ in range(batch_size)]]
	decoder_hidden = encoder_hidden
	for time_step in range(max_length):
		decoder_input = torch.tensor(next_words).to(data_manager.device)
		logits, decoder_hidden, attention_weights = model.decode(
					decoder_input, decoder_hidden, encoder_output, encoder_mask)
		logits = logits.squeeze(0)
		logits[:, data_manager.pad_id] += hypothesis_ends * 1e9
		best_toks = torch.argmax(logits, dim=-1)
		next_words = best_toks.unsqueeze(0)
		batch_hypothesis = torch.cat((batch_hypothesis, best_toks.unsqueeze(1)), dim=-1)
		hypothesis_ends = hypothesis_ends | (best_toks == data_manager.eos_id)
	return [data_manager.vocab.DecodeIds(hypothesis.cpu().numpy().tolist()) for hypothesis in batch_hypothesis]


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
						model, source_sentences[start_index:start_index + batch_size], data_manager)
			else:
				prediction_batch = predict_beam(
						model, source_sentences[start_index:start_index + batch_size], data_manager)
				prediction_batch = [candidates[0] for candidates in prediction_batch]
			predictions.extend(prediction_batch)
	return sacrebleu.corpus_bleu(predictions, [target_sentences]).score


def get_grad_influence(model, data_manager):
    assert model.dropout == 0
    model.train()
    val_data_iter = make_batch_iterator(data_manager, 1)
    influences = []
    for data_id, (source, target) in enumerate(val_data_iter):
        src_emb = model.embedding(source).squeeze(1)
        encoder_output, encoder_mask, encoder_hidden = model.encode(source)
        decoder_input, decoder_target = target[:-1], target[1:]
        logits, decoder_hidden, attention_weights = model.decode(
            decoder_input, encoder_hidden, encoder_output, encoder_mask)
        logits = logits.squeeze()
        decoder_target = decoder_target.squeeze()
        tgt_length = torch.sum(decoder_target.squeeze() != data_manager.pad_id).item()

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

def get_grad_influence2(model, data_manager, bsize):
    assert model.dropout == 0
    model.train()
    val_data_iter = make_batch_iterator(data_manager, bsize)
    influences = []
    for data_id, (source, target) in enumerate(val_data_iter):
        encoder_output, encoder_mask, encoder_hidden = model.encode(source, in_grad=True)
        decoder_input, decoder_target = target[:-1], target[1:]
        logits, decoder_hidden, attention_weights = model.decode(
            decoder_input, encoder_hidden, encoder_output, encoder_mask)

        grads = []
        for i in range(decoder_target.shape[0]):
            grad = model.influence(-logits[i, list(range(len(decoder_target[i]))), decoder_target[i]], retain_graph=(i != decoder_target.shape[0] - 1))
            grads.append(grad)
        grads = torch.stack(grads, dim=0).permute(2, 0, 1)
        for idx in range(len(grads)):
        	tgt_length = torch.sum(decoder_target[:, idx].squeeze() != data_manager.pad_id).item()
        	src_length = torch.sum(source[:, idx].squeeze() != data_manager.pad_id).item()
        	influences.append(-grads[idx][:tgt_length, :src_length].clone().detach().cpu().numpy())
    return influences


