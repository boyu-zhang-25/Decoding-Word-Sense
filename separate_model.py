import torch
import pickle
from graph_lstm import *
from decoder import *
from graph2seq_model import *

# separate the decoder and grapsh from pretraining graph
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

	graph2seq_model.load_state_dict(torch.load(path))
	# print(graph2seq_model.decoder.state_dict())

	# separate the decoder
	torch.save(graph2seq_model.decoder.state_dict(), './models/decoder.pth')

	# separate the two graph lstms
	torch.save(graph2seq_model.hyper_hypon_graph.state_dict(), './models/hyper_hypon_graph.pth')
	torch.save(graph2seq_model.mer_holo_graph.state_dict(), './models/mer_holo_graph.pth')	

separate_from_graph('./models/graph2seq_best_model.pth')
