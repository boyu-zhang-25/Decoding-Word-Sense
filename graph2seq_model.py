import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Iterable, defaultdict
import itertools
import random
from graph_lstm import *
from decoder import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
graph_lstm-decoder
'''
class Graph2Seq_Model(nn.Module):

	def __init__(self,
				hyper_hypon_graph,
				mer_holo_graph,
				hyper_hypon_depth,
				mer_holo_depth, 
				decoder,
				vocab,
				max_seq_length,
				decoder_hidden_size,
				word_embed_size = 256,
				dropout = 0.2, 
				regularization = None,
				device = device):
		super(Graph2Seq_Model, self).__init__()
		self.device = device

		# the word embedding size for the nn.Embedding
		# NOTICE: 1/2 of the decoder.input_size because we will concat later
		self.word_embed_size = word_embed_size

		'''
		if regularization == "l1":
			self.regularization = L1Loss()
		elif regularization == "smoothl1":
			self.regularization = SmoothL1Loss()
		else:
			self.regularization = None
		'''
		'''
		if self.regularization:
			self.regularization = self.regularization.to(self.device)
		'''

		# sense embedding is made by concat [[hyper, hypon], [mer, holo]]
		self.hyper_hypon_graph = hyper_hypon_graph
		self.mer_holo_graph = mer_holo_graph
		self.hyper_hypon_depth = hyper_hypon_depth
		self.mer_holo_depth = mer_holo_depth

		# decoder and its settings
		self.decoder = decoder
		self.max_length = max_seq_length
		self.vocab_size = vocab.idx
		self.decoder_hidden_size = decoder_hidden_size

		# word embedding for decoding sense
		self.pad_idx = vocab('<pad>')
		self.start_idx = vocab('<start>')
		# self.end_idx = vocab('<end>')
		self.embed = nn.Embedding(vocab.idx, self.word_embed_size, padding_idx = self.pad_idx)
		self.dropout = nn.Dropout(dropout)

	# perform per-synset SGD WSD 
	# as a pretrain model
	def forward(self, synset, definition, teacher_forcing_ratio = 0.4):
		
		'''
		teacher_forcing: the probability of using ground truth in decoding

		definition: a list 
		row as the batch ( == 1 here) and column as words: indices of each word in the true definition from the vocab

		synset: name of the target synset node
		'''
		
		# SGD per synset node on the graph
		batch_size = 1

		# set the graph_lstm output same size as the embedding for now
		# check size compatible with the decoder input
		# assert(self.graph_lstm.hidden_size * self.graph_lstm.num_layers == self.word_embed_size)
		# assert(self.graph_lstm.hidden_size * self.graph_lstm.num_layers + self.word_embed_size == self.decoder.input_size)

		# sense embedding from the graph_lstm runing on the target synset node
		# hyper_hypon_graph
		hyper_hypon_h_all, (hyper_hypon_hidden, hyper_hypon_cell) = self.hyper_hypon_graph(synset, depth = self.hyper_hypon_depth)

		# mer_holo_graph
		mer_holo_h_all, (mer_holo_hidden, mer_holo_cell) = self.mer_holo_graph(synset, depth = self.mer_holo_depth)

		# sense embedding is made by concat [[hyper, hypon], [mer, holo]]
		graph_lstm_embedding = torch.cat((hyper_hypon_hidden, mer_holo_hidden)).unsqueeze(0)
		# print('graph out size: {}'.format(graph_lstm_embedding.shape))

		# tensor to store decoder outputs
		outputs = torch.zeros(self.max_length, batch_size, self.vocab_size).to(self.device)

		# initialize the x_0 with <start>
		# (batch_size == 1, word_embed_size)
		lookup_tensor = torch.tensor([self.start_idx], dtype = torch.long).to(self.device)
		generated_embedding = self.dropout(self.embed(lookup_tensor)).to(self.device)
		# print(generated_embedding.shape)

		# concat of the graph_lstm embedding and the generated word embedding
		# (batch_size == 1, decoder.input_size)
		sense_embedding = torch.cat((graph_lstm_embedding, generated_embedding), 1).to(self.device)

		# initialize h_0 and c_0 for the decoder LSTM_Cell as the h and c of the graph encoder
		'''
		hidden = torch.zeros(batch_size, self.decoder_hidden_size).to(self.device)
		cell = torch.zeros(batch_size, self.decoder_hidden_size).to(self.device)
		'''
		hidden = graph_lstm_embedding
		cell = torch.cat((hyper_hypon_cell, mer_holo_cell)).unsqueeze(0)

		# visualize the result
		result = []

		# explicitly iterate through the max_length to decode
		for t in range(self.max_length):
			
			# get embedding at each time step from the LSTM
			# (batch, vocab_size), (batch, decoder.hidden_size)
			output, hidden, cell = self.decoder(sense_embedding, hidden, cell)
			# print('deocder out size in model: {}'.format(output.shape))
			# print(output.shape)
			outputs[t] = output

			# get the max word index from the vocabulary
			_, generated_index = torch.max(output, dim = 1)
			# print(generated_index)
			result.append(generated_index)

			# final word choice
			word_index = -1

			# get the generated word index for the target synset
			teacher_force = random.random() < teacher_forcing_ratio
			if teacher_force:
				word_index = definition[t]
			else:
				word_index = generated_index

			# get the new embedding
			# concat the graph_lstm embedding to the generated embedding at each time step
			lookup_tensor = torch.tensor([word_index], dtype = torch.long).to(self.device)
			generated_embedding = self.dropout(self.embed(lookup_tensor)).to(self.device)
			# print(generated_embedding.shape)			
			sense_embedding = torch.cat((graph_lstm_embedding, generated_embedding), 1).to(self.device)

		# print('model output size: {}'.format(outputs.shape))
		return outputs, result
