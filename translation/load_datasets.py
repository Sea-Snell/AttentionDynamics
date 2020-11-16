import torchtext
import sentencepiece
import os
from typing import List
import torch
import torch.nn as nn
import random

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


def load_europarl9_cs_en():
        with open('data/europarl9-cs-en/train.cs', 'r') as f:
                train_cs = f.read().split('\n')
        with open('data/europarl9-cs-en/train.en', 'r') as f:
                train_en = f.read().split('\n')
        with open('data/europarl9-cs-en/val.cs', 'r') as f:
                val_cs = f.read().split('\n')
        with open('data/europarl9-cs-en/val.en', 'r') as f:
                val_en = f.read().split('\n')
        
        training_data = [Item(train_cs[i], train_en[i]) for i in range(len(train_en))]
        validation_data = [Item(val_cs[i], val_en[i]) for i in range(len(val_en))]
        vocab = init_vocab("data/europarl9-cs-en/train.cs,data/europarl9-cs-en/train.en", "data/europarl9-cs-en/europarl9-cs-en")
        return training_data, validation_data, vocab

def load_europarl9_it_en():
        with open('data/europarl9-it-en/train.it', 'r') as f:
                train_it = f.read().split('\n')
        with open('data/europarl9-it-en/train.en', 'r') as f:
                train_en = f.read().split('\n')
        with open('data/europarl9-it-en/val.it', 'r') as f:
                val_it = f.read().split('\n')
        with open('data/europarl9-it-en/val.en', 'r') as f:
                val_en = f.read().split('\n')
        
        training_data = [Item(train_it[i], train_en[i]) for i in range(len(train_en))]
        validation_data = [Item(val_it[i], val_en[i]) for i in range(len(val_en))]
        vocab = init_vocab("data/europarl9-it-en/train.it,data/europarl9-it-en/train.en", "data/europarl9-it-en/europarl9-it-en")
        return training_data, validation_data, vocab

def load_news_commentary_v14_ar_en():
        with open('data/news-commentary-v14-ar-en/train.ar', 'r') as f:
                train_ar = f.read().split('\n')
        with open('data/news-commentary-v14-ar-en/train.en', 'r') as f:
                train_en = f.read().split('\n')
        with open('data/news-commentary-v14-ar-en/val.ar', 'r') as f:
                val_ar = f.read().split('\n')
        with open('data/news-commentary-v14-ar-en/val.en', 'r') as f:
                val_en = f.read().split('\n')
        
        training_data = [Item(train_ar[i], train_en[i]) for i in range(len(train_en))]
        validation_data = [Item(val_ar[i], val_en[i]) for i in range(len(val_en))]
        vocab = init_vocab("data/news-commentary-v14-ar-en/train.ar,data/news-commentary-v14-ar-en/train.en", "data/news-commentary-v14-ar-en/news-commentary-v14-ar-en")
        return training_data, validation_data, vocab

def load_news_commentary_v14_cs_de():
        with open('data/news-commentary-v14-cs-de/train.cs', 'r') as f:
                train_cs = f.read().split('\n')
        with open('data/news-commentary-v14-cs-de/train.de', 'r') as f:
                train_de = f.read().split('\n')
        with open('data/news-commentary-v14-cs-de/val.cs', 'r') as f:
                val_cs = f.read().split('\n')
        with open('data/news-commentary-v14-cs-de/val.de', 'r') as f:
                val_de = f.read().split('\n')
        
        training_data = [Item(train_cs[i], train_de[i]) for i in range(len(train_de))]
        validation_data = [Item(val_cs[i], val_de[i]) for i in range(len(val_de))]
        vocab = init_vocab("data/news-commentary-v14-cs-de/train.cs,data/news-commentary-v14-cs-de/train.de", "data/news-commentary-v14-cs-de/news-commentary-v14-cs-de")
        return training_data, validation_data, vocab

def load_news_commentary_v14_cs_fr():
        with open('data/news-commentary-v14-cs-fr/train.cs', 'r') as f:
                train_cs = f.read().split('\n')
        with open('data/news-commentary-v14-cs-fr/train.fr', 'r') as f:
                train_fr = f.read().split('\n')
        with open('data/news-commentary-v14-cs-fr/val.cs', 'r') as f:
                val_cs = f.read().split('\n')
        with open('data/news-commentary-v14-cs-fr/val.fr', 'r') as f:
                val_fr = f.read().split('\n')
        
        training_data = [Item(train_cs[i], train_fr[i]) for i in range(len(train_fr))]
        validation_data = [Item(val_cs[i], val_fr[i]) for i in range(len(val_fr))]
        vocab = init_vocab("data/news-commentary-v14-cs-fr/train.cs,data/news-commentary-v14-cs-fr/train.fr", "data/news-commentary-v14-cs-fr/news-commentary-v14-cs-fr")
        return training_data, validation_data, vocab

# def load_news_commentary_v14_en_hi():
#       with open('data/news-commentary-v14-en-hi/train.en', 'r') as f:
#               train_en = f.read().split('\n')
#       with open('data/news-commentary-v14-en-hi/train.hi', 'r') as f:
#               train_hi = f.read().split('\n')
#       with open('data/news-commentary-v14-en-hi/val.en', 'r') as f:
#               val_en = f.read().split('\n')
#       with open('data/news-commentary-v14-en-hi/val.hi', 'r') as f:
#               val_hi = f.read().split('\n')

#       training_data = [Item(train_en[i], train_hi[i]) for i in range(len(train_hi))]
#       validation_data = [Item(val_en[i], val_hi[i]) for i in range(len(val_hi))]
#       vocab = init_vocab("data/news-commentary-v14-en-hi/train.en,data/news-commentary-v14-en-hi/train.hi", "data/news-commentary-v14-en-hi/news-commentary-v14-en-hi")
#       return training_data, validation_data, vocab

def load_news_commentary_v14_ar_id():
        with open('data/news-commentary-v14-ar-id/train.ar', 'r') as f:
                train_ar = f.read().split('\n')
        with open('data/news-commentary-v14-ar-id/train.id', 'r') as f:
                train_id = f.read().split('\n')
        with open('data/news-commentary-v14-ar-id/val.ar', 'r') as f:
                val_ar = f.read().split('\n')
        with open('data/news-commentary-v14-ar-id/val.id', 'r') as f:
                val_id = f.read().split('\n')
        
        training_data = [Item(train_ar[i], train_id[i]) for i in range(len(train_id))]
        validation_data = [Item(val_ar[i], val_id[i]) for i in range(len(val_id))]
        vocab = init_vocab("data/news-commentary-v14-ar-id/train.ar,data/news-commentary-v14-ar-id/train.id", "data/news-commentary-v14-ar-id/news-commentary-v14-ar-id")
        return training_data, validation_data, vocab

def load_news_commentary_v14_en_nl():
        with open('data/news-commentary-v14-en-nl/train.en', 'r') as f:
                train_en = f.read().split('\n')
        with open('data/news-commentary-v14-en-nl/train.nl', 'r') as f:
                train_nl = f.read().split('\n')
        with open('data/news-commentary-v14-en-nl/val.en', 'r') as f:
                val_en = f.read().split('\n')
        with open('data/news-commentary-v14-en-nl/val.nl', 'r') as f:
                val_nl = f.read().split('\n')

        training_data = [Item(train_en[i], train_nl[i]) for i in range(len(train_nl))]
        validation_data = [Item(val_en[i], val_nl[i]) for i in range(len(val_nl))]
        vocab = init_vocab("data/news-commentary-v14-en-nl/train.en,data/news-commentary-v14-en-nl/train.nl", "data/news-commentary-v14-en-nl/news-commentary-v14-en-nl")
        return training_data, validation_data, vocab

def load_news_commentary_v14_ar_hi():
        with open('data/news-commentary-v14-ar-hi/train.ar', 'r') as f:
                train_ar = f.read().split('\n')
        with open('data/news-commentary-v14-ar-hi/train.hi', 'r') as f:
                train_hi = f.read().split('\n')
        with open('data/news-commentary-v14-ar-hi/val.ar', 'r') as f:
                val_ar = f.read().split('\n')
        with open('data/news-commentary-v14-ar-hi/val.hi', 'r') as f:
                val_hi = f.read().split('\n')
        
        training_data = [Item(train_ar[i], train_hi[i]) for i in range(len(train_hi))]
        validation_data = [Item(val_ar[i], val_hi[i]) for i in range(len(val_hi))]
        vocab = init_vocab("data/news-commentary-v14-ar-hi/train.ar,data/news-commentary-v14-ar-hi/train.hi", "data/news-commentary-v14-ar-hi/news-commentary-v14-ar-hi")
        return training_data, validation_data, vocab


def load_dataset_by_name(dataset):
        if dataset == 'multi30k':
                return load_multi30k()
        if dataset == 'iwslt14':
                return load_iwslt14()
        if dataset == 'europarl9_cs_en':
                return load_europarl9_cs_en()
        if dataset == 'europarl9_it_en':
                return load_europarl9_it_en()
        if dataset == 'news_commentary_v14_ar_en':
                return load_news_commentary_v14_ar_en()
        if dataset == 'load_news_commentary_v14_cs_de':
                return load_news_commentary_v14_cs_de()
        if dataset == 'news_commentary_v14_cs_fr':
                return load_news_commentary_v14_cs_fr()
        # if dataset == 'news_commentary_v14_en_hi':
        #       return load_news_commentary_v14_en_hi()
        if dataset == 'news_commentary_v14_ar_id':
                return load_news_commentary_v14_ar_id()
        if dataset == 'news_commentary_v14_ar_hi':
                return load_news_commentary_v14_ar_hi()
        if dataset == 'news_commentary_v14_en_nl':
                return load_news_commentary_v14_en_nl()


