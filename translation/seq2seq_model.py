from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import numpy as np


class Seq2seq(nn.Module):
	def __init__(self, device, vocab_size, hidden_dim, dropout, num_layers=2, bidirectional=True, attn_lambda=False, pad_id=-1, full_model=False, invasive_uniform=False):
		super(Seq2seq, self).__init__()
		self.hidden_dim, self.dropout = hidden_dim, dropout
		self.embedding = nn.Embedding(vocab_size, hidden_dim)
		self.num_layers = num_layers
		self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim,
		                            num_layers=self.num_layers, dropout=dropout, bidirectional=bidirectional)
		self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim,
		                            num_layers=self.num_layers, dropout=dropout, bidirectional=False)
		self.extra_ff = nn.Linear(2 * hidden_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.hidden2vocab = nn.Linear(hidden_dim, vocab_size)
		self.hidden_dim_scale = 2 if bidirectional else 1
		self.vocab_size = vocab_size
		self.lsm = nn.LogSoftmax(dim=-1)
		self.pad_id = pad_id
		self.loss = torch.nn.NLLLoss(ignore_index=self.pad_id)
		self.attn_lambda = attn_lambda
		self.attn_lin = nn.Linear(hidden_dim, self.hidden_dim_scale * hidden_dim)
		self.project_enc = nn.Linear(self.hidden_dim_scale * hidden_dim, hidden_dim)
		self.sm = nn.Softmax(dim=-1)
		self.attn_lambda = attn_lambda
		self.grads = {}

		self.attn_param_group = []
		self.attn_param_group.extend(list(self.attn_lin.parameters()))

		self.model_param_group = []
		self.model_param_group.extend(list(self.embedding.parameters()))
		self.model_param_group.extend(list(self.encoder_lstm.parameters()))
		self.model_param_group.extend(list(self.decoder_lstm.parameters()))
		self.model_param_group.extend(list(self.hidden2vocab.parameters()))
		self.model_param_group.extend(list(self.project_enc.parameters()))
		self.full_model = full_model
		self.invasive_uniform = invasive_uniform
		self.device = device
		self.inputs = None



	def save_grad(self, name):
		def hook(grad):
			self.grads[name] = grad
		return hook

	def convert_hidden(self, hidden):
		batch_first_hidden = hidden.transpose(1, 0)
		batch_size = batch_first_hidden.shape[0]
		batch_lyr_direction_hidden = batch_first_hidden.reshape(batch_size, self.num_layers, self.hidden_dim_scale, self.hidden_dim)
		batch_first_avg_direction = torch.mean(batch_lyr_direction_hidden, dim=2)
		assert batch_first_avg_direction.shape == (batch_size, self.num_layers, self.hidden_dim)
		return batch_first_avg_direction.transpose(1, 0).contiguous()

	def encode(self, source, in_grad=False):
		lengths = torch.sum(source != self.pad_id, axis=0).cpu()
		encoder_mask = (source == self.pad_id)
		source_emb = self.embedding(source)
		if in_grad:
                    del self.inputs
                    self.inputs = torch.autograd.Variable(torch.ones(source.size()).to(self.device), requires_grad=True)
                    source_emb = (source_emb.permute(2, 0, 1) * self.inputs).permute(1, 2, 0)
		else:
                    del self.inputs
                    self.inputs = None

		if self.training:
                    source_emb.register_hook(self.save_grad('source_emb'))
		packed_input = pack_padded_sequence(source_emb, lengths, enforce_sorted=False)
		packed_output, (h_n, c_n) = self.encoder_lstm(packed_input)
		encoder_output, _ = pad_packed_sequence(packed_output)
		encoder_hidden = (self.convert_hidden(h_n), self.convert_hidden(c_n))
		# return encoder_output, encoder_mask, (encoder_hidden, (torch.unsqueeze(torch.mean(encoder_hidden[0], 0), 0), torch.unsqueeze(torch.mean(encoder_hidden[1], 0), 0)))
		return encoder_output, encoder_mask, (encoder_hidden, (torch.zeros([1] + list(encoder_hidden[0].shape)[1:-1] + [2 * encoder_hidden[0].shape[-1]]).to(self.device), torch.zeros([1] + list(encoder_hidden[1].shape)[1:-1] + [2 * encoder_hidden[1].shape[-1]]).to(self.device)))

	def compute_loss(self, source, target, ref_attn_func=None):
		"""Run the model on the source and compute the loss on the target.

		Args:
		  source: An integer tensor with shape (max_source_sequence_length,
		    batch_size) containing subword indices for the source sentences.
		  target: An integer tensor with shape (max_target_sequence_length,
		    batch_size) containing subword indices for the target sentences.

		Returns:
		  A scalar float tensor representing cross-entropy loss on the current batch.
		"""

		# Implementation tip: don't feed the target tensor directly to the decoder.
		# To see why, note that for a target sequence like <s> A B C </s>, you would
		# want to run the decoder on the prefix <s> A B C and have it predict the
		# suffix A B C </s>.

		# YOUR CODE HERE
		encoder_output, encoder_mask, encoder_hidden = self.encode(source)
		target_in, target_out = target[:-1], target[1:]
		logits, decoder_hidden, attention = self.decode(target_in, encoder_hidden,
		                                                encoder_output, encoder_mask)
		seq_len, b_size, _ = logits.shape
		num_toks = seq_len * b_size
		loss = self.loss(logits.reshape(num_toks, self.vocab_size), target_out.reshape(num_toks))
		clf_loss = loss.item()
		if ref_attn_func is not None and self.attn_lambda != 0:
			target_attn = ref_attn_func(source, target)
			assert attention.shape == target_attn.shape
			entry_wise_diff = torch.abs(attention - target_attn)
			loss += self.attn_lambda * torch.sum(entry_wise_diff) / torch.sum(encoder_mask == False)
		return {
		    'loss': loss,
		    'clf': clf_loss,
		    'attn': loss - clf_loss,
		    'attention': attention
		}

	def decode(self, decoder_input, initial_hidden, encoder_output, encoder_mask):
		"""Run the decoder LSTM starting from an initial hidden state.

		The third and fourth arguments are not used in the baseline model, but are
		included for compatibility with the attention model in the next section.

		Args:
		  decoder_input: An integer tensor with shape (max_decoder_sequence_length,
		    batch_size) containing the subword indices for the decoder input. During
		    evaluation, where decoding proceeds one step at a time, the initial
		    dimension should be 1.
		  initial_hidden: A pair of tensors (h_0, c_0) representing the initial
		    state of the decoder, each with shape (num_layers, batch_size,
		    hidden_size).
		  encoder_output: The output of the encoder with shape
		    (max_source_sequence_length, batch_size, 2 * hidden_size).
		  encoder_mask: The output mask from the encoder with shape
		    (max_source_sequence_length, batch_size). Encoder outputs at positions
		    with a True value correspond to padding tokens and should be ignored.

		Returns:
		  A tuple with three elements:
		    logits: A tensor with shape (max_decoder_sequence_length, batch_size,
		      vocab_size) containing unnormalized scores for the next-word
		      predictions at each position.
		    decoder_hidden: A pair of tensors (h_n, c_n) with the same shape as
		      initial_hidden representing the updated decoder state after processing
		      the decoder input.
		    attention_weights: A tensor with shape (max_decoder_sequence_length,
		      batch_size, max_source_sequence_length) representing the normalized
		      attention weights. This should sum to 1 along the last dimension.
		"""

		# Implementation tip: use a large negative number like -1e9 instead of
		# float("-inf") when masking logits to avoid numerical issues.

		# Implementation tip: the function torch.einsum may be useful here.
		# See https://rockt.github.io/2018/04/30/einsum for a tutorial.

		# YOUR CODE HERE
		tgt_length, batch_size = decoder_input.shape
		decoder_emb = self.embedding(decoder_input)
		initial_hidden1, initial_hidden2 = initial_hidden
		decoder_output, decoder_hidden = self.decoder_lstm(decoder_emb, initial_hidden1)
		tgt_bsize_feat_query = self.attn_lin(decoder_output)
		tgt_bsize_src_scores = torch.einsum("tbf,sbf->tbs", [tgt_bsize_feat_query, encoder_output])
		if self.invasive_uniform:
			tgt_bsize_src_scores = torch.zeros_like(tgt_bsize_src_scores)

		tgt_bsize_src_mask = encoder_mask.unsqueeze(0).repeat(tgt_length, 1, 1).permute(0, 2, 1)
		tgt_bsize_src_scores += tgt_bsize_src_mask * (-1e9)
		attn = self.sm(tgt_bsize_src_scores)

		weighted_memory = torch.einsum("tbs,sbf->tbf", [attn, encoder_output])
		assert weighted_memory.shape == (tgt_length, batch_size, self.hidden_dim_scale * self.hidden_dim)
		c = self.project_enc(weighted_memory)
		decoder_hidden2 = None
		if not self.full_model:
			output = decoder_output + c
		else:
			cat_output = torch.cat((decoder_output, c), dim=-1)
			decoder_hidden2 = initial_hidden2
			output = self.relu(self.extra_ff(cat_output))
		logits = self.lsm(self.hidden2vocab(output))
		return logits, (decoder_hidden, decoder_hidden2), attn

	def get_state_scores(self, source, target):
		encoder_output, encoder_mask, encoder_hidden = self.encode(source)
		target_in, target_out = target[:-1], target[1:]
		bsize_src_vocab = self.sm(self.hidden2vocab(self.project_enc(encoder_output))).detach().cpu().numpy()
		bsize = bsize_src_vocab.shape[1]
		src_lengths = torch.sum(source != self.pad_id, axis=0).cpu()
		tgt_lengths = torch.sum(target_out != self.pad_id, axis=0).cpu()
		batch_state_scores = []
		for data_id in range(bsize):
			src_length, tgt_length = src_lengths[data_id], tgt_lengths[data_id]
			state_scores = []
			for tgt_id in range(tgt_length):
				state_probs = bsize_src_vocab[:src_length, data_id, target_out[tgt_id][data_id]]
				# state_scores.append(np.log(state_probs / (1 - state_probs)))
				state_scores.append(np.log(state_probs))
			state_scores = np.array(state_scores)
			batch_state_scores.append(state_scores)
		return batch_state_scores

  #handles non-linearity approximation
	def get_state_scores2(self, source, target):
		encoder_output, encoder_mask, encoder_hidden = self.encode(source)
		bsize = encoder_output.shape[1]

		target_in, target_out = target[:-1], target[1:]
		decoder_emb = self.embedding(target_in)
		decoder_output, decoder_hidden = self.decoder_lstm(decoder_emb, encoder_hidden[0])

		c = self.project_enc(encoder_output)
		tile_c = c.unsqueeze(dim=0).repeat(decoder_output.shape[0], *[1]*len(encoder_output.shape))
		tile_decoder_output = decoder_output.unsqueeze(dim=1).repeat(1, encoder_output.shape[0], *[1]*(len(decoder_output.shape) - 1))

		cat_output = torch.cat((tile_decoder_output, tile_c), dim=-1)
		output = self.relu(self.extra_ff(cat_output))
		logits = self.hidden2vocab(output)
		probs = self.sm(logits)

		src_lengths = torch.sum(source != self.pad_id, axis=0).cpu()
		tgt_lengths = torch.sum(target_out != self.pad_id, axis=0).cpu()
		batch_state_scores = []
		for data_id in range(bsize):
			src_length, tgt_length = src_lengths[data_id], tgt_lengths[data_id]
			state_scores = []
			for tgt_id in range(tgt_length):
				scores = probs[tgt_id, :src_length, data_id, target_out[tgt_id][data_id]].clone().detach().cpu().numpy()
				# state_scores.append(np.log(scores / (1 - scores)))
				state_scores.append(np.log(scores))
			state_scores = np.array(state_scores)
			batch_state_scores.append(state_scores)
		return batch_state_scores


	def influence(self, logits, retain_graph):
		self.zero_grad()
		if self.inputs.grad is not None:
			self.inputs.grad.data.zero_()
		torch.sum(logits).backward(retain_graph=retain_graph)
		grad = torch.tensor(self.inputs.grad)
		self.zero_grad()
		self.inputs.grad.data.zero_()
		return grad





