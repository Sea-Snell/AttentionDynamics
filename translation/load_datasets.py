import torchtext
import sentencepiece
import os
from typing import List
import torch
import torch.nn as nn
import random
import json

class StateManager:
    def __init__(self, dataset, vocab, bos_id, eos_id, pad_id, device, config):
        self.vocab = vocab
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.dataset = dataset
        self.device = device
        self.pad_id = pad_id
        self.vocab_size = self.vocab.GetPieceSize()
        for key_ in config:
            setattr(self, key_, config[key_])

def sentence2ids_nopad(state_manager, sentence: str, additional_eos: bool) -> List[int]:
    indices = [state_manager.bos_id] + state_manager.vocab.EncodeAsIds(sentence) + [state_manager.eos_id]
    if additional_eos:
        indices.append(state_manager.eos_id)
    return indices

def make_batch(state_manager, sentences: List[str], additional_eos: bool) -> torch.Tensor:
    """Convert a list of sentences into a batch of subword indices.

    Args:
            sentences: A list of sentences, each of which is a string.

    Returns:
            A LongTensor of size (max_sequence_length, batch_size) containing the
            subword indices for the sentences, where max_sequence_length is the length
            of the longest sentence as encoded by the subword vocabulary and batch_size
            is the number of sentences in the batch. A beginning-of-sentence token
            should be included before each sequence, and an end-of-sentence token should
            be included after each sequence. Empty slots at the end of shorter sequences
            should be filled with padding tokens. The tensor should be located on the
            device defined at the beginning of the notebook.
    """

    # Implementation tip: You can use the nn.utils.rnn.pad_sequence utility
    # function to combine a list of variable-length sequences with padding.
    # YOUR CODE HERE
    #...
    batch_ids = [torch.tensor(sentence2ids_nopad(state_manager, sentence, additional_eos)) for sentence in sentences]
    return nn.utils.rnn.pad_sequence(batch_ids).to(state_manager.device)


def make_batch_iterator(state_manager, batch_size, shuffle=False):
    """Make a batch iterator that yields source-target pairs.

    Args:
            dataset: A torchtext dataset object.
            batch_size: An integer batch size.
            shuffle: A boolean indicating whether to shuffle the examples.

    Yields:
            Pairs of tensors constructed by calling the make_batch function on the
            source and target sentences in the current group of examples. The max
            sequence length can differ between the source and target tensor, but the
            batch size will be the same. The final batch may be smaller than the given
            batch size.
    """

    examples = list(state_manager.dataset)
    if shuffle:
        random.shuffle(examples)

    for start_index in range(0, len(examples), batch_size):
        example_batch = examples[start_index:start_index + batch_size]
        source_sentences = [example.src for example in example_batch]
        target_sentences = [example.trg for example in example_batch]
        yield make_batch(state_manager, source_sentences, additional_eos=False), make_batch(state_manager, target_sentences, additional_eos=False)



def init_vocab(data_files, model_prefix):
    vocab_path = model_prefix + '.model'
    if not os.path.exists(vocab_path):
        args = {
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "input": data_files,
            "vocab_size": 8000,
            "model_prefix": model_prefix,
        }
        combined_args = " ".join(
                        "--{}={}".format(key, value) for key, value in args.items())
        sentencepiece.SentencePieceTrainer.Train(combined_args)

    vocab = sentencepiece.SentencePieceProcessor()
    vocab.Load(vocab_path)

    return vocab

class Item:
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

def load_iwslt14():
    with open('data/iwslt14/train.en', 'r') as f:
        train_en = f.read().split('\n')
    with open('data/iwslt14/train.de', 'r') as f:
        train_de = f.read().split('\n')
    with open('data/iwslt14/valid.en', 'r') as f:
        val_en = f.read().split('\n')
    with open('data/iwslt14/valid.de', 'r') as f:
        val_de = f.read().split('\n')

    training_data = [Item(train_de[i], train_en[i]) for i in range(len(train_en))]
    validation_data = [Item(val_de[i], val_en[i]) for i in range(len(val_en))]
    vocab = init_vocab("data/iwslt14/train.en,data/iwslt14/train.de", "data/iwslt14/iwslt14")
    return training_data, validation_data, vocab

def load_multi30k():
    extensions = [".de", ".en"]
    source_field = torchtext.data.Field(tokenize=lambda x: x)
    target_field = torchtext.data.Field(tokenize=lambda x: x)

    training_data, validation_data, test_data = torchtext.datasets.Multi30k.splits(
                    extensions, [source_field, target_field], root="./data")
    training_data = list(training_data)
    vocab = init_vocab("data/multi30k/train.en,data/multi30k/train.de", "data/multi30k/multi30k")
    return training_data, validation_data, vocab

def load_general(folder_name):
    with open('data/%s/train.src' % (folder_name), 'r') as f:
        train_src = json.load(f)
    with open('data/%s/train.trg' % (folder_name), 'r') as f:
        train_trg = json.load(f)
    with open('data/%s/val.src' % (folder_name), 'r') as f:
        val_src = json.load(f)
    with open('data/%s/val.trg' % (folder_name), 'r') as f:
        val_trg = json.load(f)
    training_data = [Item(train_src[i], train_trg[i]) for i in range(len(train_trg))]
    validation_data = [Item(val_src[i], val_trg[i]) for i in range(len(val_trg))]
    vocab = init_vocab("data/%s/train.src_raw,data/%s/train.trg_raw" % (folder_name, folder_name), "data/%s/%s" % (folder_name, folder_name))
    return training_data, validation_data, vocab


def load_dataset_by_name(dataset):
    if dataset == 'multi30k':
        return load_multi30k()
    if dataset == 'iwslt14':
        return load_iwslt14()
    if dataset == 'news_commentary_v14_en_nl':
        return load_general('news-commentary-v14-en-nl')
    if dataset == 'news_commentary_v14_en_pt':
        return load_general('news-commentary-v14-en-pt')
    if dataset == 'news_commentary_v14_it_pt':
        return load_general('news-commentary-v14-it-pt')


