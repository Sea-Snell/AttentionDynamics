from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):

    def __init__(self, vocab_size, token_id2vector, EMBED_DIM, HIDDEN_DIM, INTERMEDIATE_DIM, device):
        super(Model, self).__init__()

        # to get the initialization
        # could be implemented more nicely
        embedding = nn.Embedding(vocab_size, EMBED_DIM)
        embedding_matrix = embedding.weight.data.numpy()
        for i in token_id2vector:
            embedding_matrix[i] = token_id2vector[i]
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix).to(device))
        self.embedding.weight.requires_grad = True
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.encoder = nn.LSTM(EMBED_DIM, hidden_size=HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.hidden2attn_intermediate = nn.Linear(2 * HIDDEN_DIM, INTERMEDIATE_DIM)
        self.attn_intermediate2logit = nn.Linear(INTERMEDIATE_DIM, 1)
        self.hidden2beta = nn.Linear(2 * HIDDEN_DIM, 1)
        self.intermediate_act = nn.ReLU()
        self.sm = nn.Softmax(dim=-1)
        self.inputs = None
        self.device = device

        self.attn_param_group = []
        self.attn_param_group.extend(list(self.hidden2attn_intermediate.parameters()))
        self.attn_param_group.extend(list(self.attn_intermediate2logit.parameters()))

        self.model_param_group = []
        self.model_param_group.extend(list(self.embedding.parameters()))
        self.model_param_group.extend(list(self.encoder.parameters()))
        self.model_param_group.extend(list(self.hidden2beta.parameters()))


    def forward(self, sentences_wout_pad, uniform, pad_token, in_grad=False):
        sentences_wout_pad = [sentence[:] for sentence in sentences_wout_pad]

        lengths = [len(sentence_wout_pad) for sentence_wout_pad in sentences_wout_pad]
        max_length = max(lengths)
        padded_sentences = [sentence + [pad_token] * (max_length- len(sentence)) for sentence in sentences_wout_pad]
        batch_length_input_tensor = torch.tensor(padded_sentences).to(self.device)
        zero_attn_mask = (batch_length_input_tensor == pad_token) * (-1e9)
        zero_beta_mask = (batch_length_input_tensor != pad_token) * 1


        embedding = self.embedding(batch_length_input_tensor)
        if in_grad:
            del self.inputs
            self.inputs = torch.autograd.Variable(torch.ones(batch_length_input_tensor.size()).to(self.device), requires_grad=True)
            embedding = (embedding.permute(2, 0, 1) * self.inputs).permute(1, 2, 0)
        else:
            del self.inputs
            self.inputs = None

        packed = pack_padded_sequence(embedding, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.encoder(packed)
        batch_length_hidden_dim_hidden_states, _ = pad_packed_sequence(packed_output, batch_first=True)

        attn_intermediate = self.intermediate_act(self.hidden2attn_intermediate(batch_length_hidden_dim_hidden_states))
        attn_logit = self.attn_intermediate2logit(attn_intermediate).squeeze(2) * (1 if not uniform else 0) + zero_attn_mask

        attn_distr = self.sm(attn_logit)
        beta = self.hidden2beta(batch_length_hidden_dim_hidden_states).squeeze(2) * zero_beta_mask
        scores = torch.sum(attn_distr * beta, axis=-1)

        # encapsulate such that padding will not leak outside
        attn_distr_numpy = [a.detach().cpu().numpy()[:lengths[i]] for i, a in enumerate(attn_distr)]
        beta_numpy = [b.detach().cpu().numpy()[:lengths[i]] for i, b in enumerate(beta)]

        return {
            'scores': scores,
            'alpha': attn_distr_numpy,
            'beta': beta_numpy,
            'lens': lengths
        }

    def influence(self, logits):
        self.zero_grad()
        if self.inputs.grad is not None:
            self.inputs.grad.data.zero_()
        torch.sum(logits).backward()
        grad = torch.tensor(self.inputs.grad)
        self.zero_grad()
        self.inputs.grad.data.zero_()
        return grad


