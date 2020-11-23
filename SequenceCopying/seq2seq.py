import numpy as np
import torch
from argparse import ArgumentParser
from seq2seq_model import Seq2seq
import random
import pickle as pkl


device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = ArgumentParser()
parser.add_argument('--num_vocab', type=int, default=40)
parser.add_argument('--seq_length', type=int, default=40)
parser.add_argument('--bi', default=False, action='store_true')
parser.add_argument('--bsize', type=int, default=128)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--steps_s1', type=int, default=200)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--no_rep', default=False, action='store_true')
parser.add_argument('--invasive_uniform', default=False, action='store_true')
parser.add_argument('--early_stop', default=False, action='store_true')
parser.add_argument('--save_every', default=10, type=int)
parser.add_argument('--no_context', default=False, action='store_true')


args = parser.parse_args()
print(args)

seq_length = args.seq_length
num_vocab = args.num_vocab
exp_name = args.exp_name
bi = args.bi
steps_s1 = args.steps_s1
bsize = args.bsize
hidden_dim = args.hidden_dim
no_rep = args.no_rep
invasive_uniform = args.invasive_uniform
early_stop = args.early_stop
no_context = args.no_context


if exp_name is None:

    exp_name = 'vocab%dseq_length%d%sbsize%d' \
               % (num_vocab, seq_length, 'bi' if bi else '', bsize)
    if no_rep:
        exp_name += 'no_rep'
    if invasive_uniform:
        exp_name += 'invasive_uniform'

print(exp_name)
eps = 1e-10

end_of_sequence = num_vocab
start_of_sequence = num_vocab + 1


metrics = {}
def register_metrics(metric_name):
	def decorator(f):
		metrics[metric_name] = f
	return decorator


@register_metrics('acc')
def acc(return_dict, X, Y):
	final_probs = return_dict['probs']
	ans = np.argmax(final_probs, axis=-1)
	return np.mean(np.equal(ans, Y[:, 1:].cpu().numpy()))


@register_metrics('attn_acc')
def attn_acc(return_dict, X, Y):
	weights = return_dict['attn_weights']
	max_attn = np.argmax(weights, axis=-1)
	gold_attn = np.array([[i for i in range(seq_length)] for _ in range(X.size()[0])])
	return np.mean(np.equal(max_attn[:, :-1], gold_attn))


@register_metrics('attn_acc_m1')
def attn_acc_m1(return_dict, X, Y):
	weights = return_dict['attn_weights']
	max_attn = np.argmax(weights, axis=-1)
	gold_attn = np.array([[i + 1 for i in range(seq_length)] for _ in range(X.size()[0])])
	return np.mean(np.equal(max_attn[:, :-1], gold_attn))


@register_metrics('attn_acc_p1')
def attn_acc_m1(return_dict, X, Y):
	weights = return_dict['attn_weights']
	max_attn = np.argmax(weights, axis=-1)
	gold_attn = np.array([[i - 1 for i in range(seq_length)] for _ in range(X.size()[0])])
	return np.mean(np.equal(max_attn[:, :-1], gold_attn))


def generate_seq(num_samples):
    X, Y = [], []
    for _ in range(num_samples):
        if not no_rep:
            x = np.random.randint(0, num_vocab, size=(seq_length,))
        else:
            arr = [_ for _ in range(num_vocab)]
            random.shuffle(arr)
            x = np.array(arr[:seq_length])
        x = np.concatenate((x, [end_of_sequence]))

        y = np.concatenate(([start_of_sequence], x))
        X.append(x)
        Y.append(y)
    return X, Y


def numpify_dict(d):
    return {k: v.cpu().detach().numpy() for k, v in d.items()}

def eval_acc(model, data_generator):
    model.eval()
    X, Y = data_generator(50)
    src_length = len(X[0])
    source, target = torch.tensor(X).to(device).T, torch.tensor(Y).to(device).T
    encoder_output, encoder_mask, encoder_hidden = model.encode(source)
    decoder_input, decoder_target = target[:-1], target[1:]
    logits, decoder_hidden, attention_weights = model.decode(
        decoder_input, encoder_hidden, encoder_output, encoder_mask)
    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    state_scores = model.get_state_scores2_one_off(source, target)
    acc = np.mean(preds == target[1:].detach().cpu().numpy())
    attention_weights = attention_weights.permute((1, 0, 2)).detach().cpu().numpy()
    attn_acc = np.mean(np.argmax(attention_weights, axis=-1) == np.repeat(np.expand_dims(np.arange(src_length), axis=0), len(X), axis=0))
    attn_acc_m1 = np.mean((np.argmax(attention_weights, axis=-1) - 1) == np.repeat(np.expand_dims(np.arange(src_length), axis=0), len(X), axis=0))
    attn_acc_p1 = np.mean((np.argmax(attention_weights, axis=-1) + 1) == np.repeat(np.expand_dims(np.arange(src_length), axis=0), len(X), axis=0))
    model.train()
    return {'attn_acc': attn_acc, 'acc': acc, 'attn_acc_m1': attn_acc_m1, 'attn_acc_p1': attn_acc_p1}


def train_loop(m, data_generator, num_steps, invasive_uniform=invasive_uniform):
    # optimizer
    optim = torch.optim.Adam(m.parameters())
    all_eval = []
    for step in range(num_steps):
        X, Y = data_generator(bsize)
        X, Y = torch.tensor(X).to(device), torch.tensor(Y).to(device)
        loss = m.compute_loss(X.T, Y.T, invasive_uniform=invasive_uniform)['loss']
        loss.backward()
        optim.step()
        optim.zero_grad()
        if step % args.save_every == 0:
            eval_dict = eval_acc(m, data_generator)
            eval_dict['step'] = step
            all_eval.append(eval_dict)
            print(eval_dict)
    return all_eval


exp_dir = 'logs/' + exp_name

all_histories = []
for seed in range(20):
    np.random.seed(seed)
    torch.manual_seed(seed)
    m = Seq2seq(hidden_dim=hidden_dim, dropout=0, vocab_size=num_vocab + 2,
                num_layers=1, bidirectional=bi, no_context=no_context).to(device)
    data_generator = generate_seq
    history = train_loop(m, data_generator=data_generator, num_steps=steps_s1, invasive_uniform=invasive_uniform)
    all_histories.append(history)

pkl.dump(all_histories, open(exp_dir + '.pkl', 'wb'))

