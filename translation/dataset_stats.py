import numpy as np
from load_datasets import load_dataset_by_name, StateManager, make_batch, make_batch_iterator, sentence2ids_nopad

dataset_name = 'news_commentary_v14_it_pt'
training_data, validation_data, vocab = load_dataset_by_name(dataset_name)
pad_id = vocab.PieceToId("<pad>")
bos_id = vocab.PieceToId("<s>")
eos_id = vocab.PieceToId("</s>")
val_data_manager = StateManager(validation_data, vocab, bos_id, eos_id, pad_id, None, {})
train_data_manager = StateManager(training_data, vocab, bos_id, eos_id, pad_id, None, {})

print('training dataset size', len(training_data))

src_lens = []
trg_lens = []
for item in train_data_manager.dataset:
	src, trg = item.src, item.trg
	src_lens.append(len(sentence2ids_nopad(train_data_manager, src, additional_eos=False)[1:-1]))
	trg_lens.append(len(sentence2ids_nopad(train_data_manager, trg, additional_eos=False)[1:-1]))

print('src median', np.median(src_lens))
print('trg median', np.median(trg_lens))