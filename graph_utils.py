import torch
import pickle
from graph_lstm import *
# from decoder import *
# from graph2seq_model import *

def get_all_hiddens(path):
	# get the graph lstm synset vocab
	with open('./data/synset_vocab.pkl', 'rb') as f:
		synset_vocab = pickle.load(f)
	print("Size of synset vocab: {}".format(synset_vocab.idx))

	# set the hyper-hypon and mer-holo graph lstms
	hyper_hypon_graph = ChildSumGraphLSTM_WordNet(
		synset_vocab = synset_vocab, 
		relationship = 'hyper_hypon', 
		input_size = 256, 
		hidden_size = 64, 
		num_layers = 2, 
		bidirectional = True, 
		bias = True, 
		dropout = 0.2)

	mer_holo_graph = ChildSumGraphLSTM_WordNet(
		synset_vocab = synset_vocab, 
		relationship = 'mer_holo',
		input_size = 256, 
		hidden_size = 64, 
		num_layers = 2, 
		bidirectional = True, 
		bias = True, 
		dropout = 0.2)

	p1 = path + 'hyper_hypon_graph.pth'
	p2 = path + 'mer_holo_graph.pth'
	hyper_hypon_graph.load_state_dict(torch.load(p1))
	mer_holo_graph.load_state_dict(torch.load(p2))

	# get the graph lstm synset vocab that only appears in the SemCor
	with open('./data/synset_vocab_SemCor.pkl', 'rb') as f:
		synset_vocab_SemCor = pickle.load(f)
	print("Size of synset vocab in SemCor: {}".format(synset_vocab_SemCor.idx))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	hyper_hypon_graph = hyper_hypon_graph.to(device)
	mer_holo_graph = mer_holo_graph.to(device)

	# all the hidden states
	hidden_embeddings = []
	final_cells = []
	hidden_hyper_hypon = []
	hidden_mer_holo = []

	for idx in range(synset_vocab_SemCor.idx):
	# for idx in range(small_size):
		
		# get the synset and definition
		synset = synset_vocab_SemCor.idx2word.get(idx).replace('__', '.')

		# sense embedding from the graph_lstm runing on the target synset node
		# hyper_hypon_graph
		hyper_hypon_h_all, (hyper_hypon_hidden, hyper_hypon_cell) = self.hyper_hypon_graph(synset, depth = 4)

		# mer_holo_graph
		mer_holo_h_all, (mer_holo_hidden, mer_holo_cell) = self.mer_holo_graph(synset, depth = 4)

		# sense embedding is made by concat [[hyper, hypon], [mer, holo]]
		hidden_embeddings.append(torch.cat((hyper_hypon_hidden, mer_holo_hidden)).unsqueeze(0))
		final_cells.append(torch.cat((hyper_hypon_cell, mer_holo_cell)).unsqueeze(0))

		hidden_hyper_hypon.append(hyper_hypon_h_all)
		hidden_mer_holo.append(mer_holo_h_all)

	with open('hidden_embeddings.pkl', 'wb') as f:
		pickle.dump(hidden_embeddings, f)
	with open('final_cells.pkl', 'wb') as f:
		pickle.dump(final_cells, f)

	with open('hidden_hyper_hypon.pkl', 'wb') as f:
		pickle.dump(hidden_hyper_hypon, f)
	with open('hidden_mer_holo.pkl', 'wb') as f:
		pickle.dump(hidden_mer_holo, f)

def separate_from_graph(path):

	# get the decoder vocab
	with open('./data/vocab.pkl', 'rb') as f:
		vocab = pickle.load(f)
		print("Size of vocab: {}".format(vocab.idx))
		
	# get the graph lstm synset vocab
	with open('./data/synset_vocab.pkl', 'rb') as f:
		synset_vocab = pickle.load(f)
	print("Size of synset vocab: {}".format(synset_vocab.idx))

	# some hyperparameters
	max_seq_length = 20
	decoder_hidden_size = 256
	decoder_input_size = 512
	mer_holo_depth = 4
	hyper_hypon_depth = 4

	# set the hyper-hypon and mer-holo graph lstms
	hyper_hypon_graph = ChildSumGraphLSTM_WordNet(
		synset_vocab = synset_vocab, 
		relationship = 'hyper_hypon', 
		input_size = 256, 
		hidden_size = 64, 
		num_layers = 2, 
		bidirectional = True, 
		bias = True, 
		dropout = 0.2)

	mer_holo_graph = ChildSumGraphLSTM_WordNet(
		synset_vocab = synset_vocab, 
		relationship = 'mer_holo',
		input_size = 256, 
		hidden_size = 64, 
		num_layers = 2, 
		bidirectional = True, 
		bias = True, 
		dropout = 0.2)

	# decoder
	# input size of decoder = 512
	# output size of per graph lstm = 2 * 64 = 128
	# concat graph embedding size = 256
	# concat word embedding size (decoder_hidden_size) = 512 = input size of decoder
	decoder = Decoder(
		vocab_size = vocab.idx, 
		max_seq_length = max_seq_length, 
		hidden_size = decoder_hidden_size, 
		input_size = decoder_input_size)

	# the model instance
	graph2seq_model = Graph2Seq_Model(
		hyper_hypon_graph,
		mer_holo_graph,
		hyper_hypon_depth,
		mer_holo_depth, 
		decoder, 
		vocab = vocab, 
		max_seq_length = max_seq_length, 
		decoder_hidden_size = decoder_hidden_size)
	# print(graph2seq_model.decoder.state_dict())

	graph2seq_model.load_state_dict(torch.load(path, map_location='cpu'))
	# print(graph2seq_model.decoder.state_dict())

	# separate the decoder
	torch.save(graph2seq_model.hyper_hypon_graph.state_dict(), './models/hyper_hypon_graph.pth')
	torch.save(graph2seq_model.mer_holo_graph.state_dict(), './models/mer_holo_graph.pth')

# separate_from_graph('./graph2seq_best_model.pth')
get_all_hiddens('./models/')