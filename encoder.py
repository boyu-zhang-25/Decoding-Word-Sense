import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Iterable, defaultdict
import itertools
from nltk.corpus import wordnet as wn

'''
The specific Synset method is lexname, e.g. wn.synsets('spring')[0].lexname(). 
That should make it really easy to get the suspersenses.
And if you have the synset name–e.g. 'spring.n.01'
you can access the supersense directly: wn.synset('spring.n.01').lexname().
Which returns 'noun.time'.
And wn.synset('spring.n.02').lexname() returns 'noun.artifact'
'''

'''
the encoder to encode the target word sense in the given context
'''
class Encoder(nn.Module):
	def __init__(self, 
				all_senses = None,
				output_size = 300, # output size of each sense vector [300, 1]
				embedding_size = 1024, # ELMo embedding size
				elmo_class = None,
				tuned_embed_size = 256,
				mlp_dropout = 0,
				lstm_hidden_size = 256,
				MLP_sizes = [512], # 1 hidden layer for fine-tuning sense vector
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
		super().__init__()
		
		# all senses for all words
		# useful for all purposes
		self.all_senses = all_senses
		self.elmo_class = elmo_class
		self.device = device

		# for dimension reduction 
		self.tuned_embed_size = tuned_embed_size 
		self.embedding_size = embedding_size

		# sizes of the fine-tuning MLP and LSTM
		self.MLP_sizes = MLP_sizes 
		self.output_size = output_size
		self.lstm_hidden_size = lstm_hidden_size

		## initialize fine-tuning MLP layers
		self.layers = nn.ModuleDict()
		self.mlp_dropout = nn.Dropout(mlp_dropout) 

		# dimension reduction for elmo
		# 3 * 1024 ELMo -> 1 * 256 
		self.dimension_reduction_MLP = nn.Linear(self.embedding_size * 3, self.tuned_embed_size).to(self.device)

		# construct a LSTM on top of ELMo
		self.lstm = nn.LSTM(self.tuned_embed_size, self.lstm_hidden_size, num_layers = 2, bidirectional = True).to(self.device)

		# build a 2-layer MLP on top of LSTM for fine-tuning
		self._init_MLP(self.tuned_embed_size * 2, self.MLP_sizes, self.output_size, param = "word_sense")


	def _init_MLP(self, input_size, hidden_sizes, output_size, param = None):
		'''
		Initialize a 2-layer MLP on top of ELMo
		w1: input_size * hidden_sizes[0]
		w2: hidden_sizes[0] * output_size
		'''

		# dict for fine-tuning MLP structures
		self.layers[param] = nn.ModuleList()

		# initialize MLP
		for h in hidden_sizes:

			layer = torch.nn.Linear(input_size, h)
			layer = layer.to(self.device)

			# append to the fine-tuning MLP
			self.layers[param].append(layer)
			# update dimension
			input_size = h

			# ReLU activation after linear layer
			layer = nn.ReLU()
			layer = layer.to(self.device)
			self.layers[param].append(layer)            

		output_layer = torch.nn.Linear(input_size, output_size)
		output_layer = output_layer.to(self.device)
		self.layers[param].append(output_layer)
		
	def _get_embedding(self, sentence):
		'''
			@param: a list of words (a sentence)

			@return: 
			ELMo embeddins of the sentence (for all three ELMo layers each 1024 dim)
			concatenates them to make a 3072 dim embedding vector
			reduces the dimension to a lower number (256)
		''' 

		# get ELMo embedding of the sentence
		# torch.Size([3 (layers), sentence length, 1024 (word vector length)])
		embedding = torch.from_numpy(self.elmo_class.embed_sentence(sentence))
		# print('119: {}'.format(embedding.requires_grad))

		# pass to CUDA
		embedding = embedding.to(self.device)

		# old: [3 (layers), sentence_length, 1024 (word vector length)]
		# new: [sentence_length, 3 (layers), word_embedding_size]
		embedding = embedding.permute(1, 0, 2)
		# print('126: {}'.format(embedding.requires_grad))

		# concatenate 3 layers and reshape
		# [sentence_length, batch_size, 3 * 1024]
		batch_size = 1
		sentence_length = embedding.size()[0]
		embedding = embedding.contiguous().view(sentence_length, batch_size, -1)
		# print('132: {}'.format(embedding.requires_grad))
		
		# [sentence_length, batch_size, 256]
		embedding = self._tune_embeddings(embedding)
		# print('139: {}'.format(embedding.requires_grad))
		# print('em require: {}'.format(embedding.requires_grad))

		return embedding


	# forward propagation selected sentence and definitions
	def forward(self, sentence, word_idx):
		
		# preserve word lemma for future use
		word_lemma = '____' + sentence[word_idx]

		# get the dimension-reduced ELMo embedding
		embedding = self._get_embedding(sentence)
		# print('em2 require: {}'.format(embedding.requires_grad))
		# print(embedding)

		# Run a Bi-LSTM and get the sense embedding
		# (seq_len, batch, num_directions * hidden_size)
		self.lstm.flatten_parameters()
		embedding_new, (hn, cn) = self.lstm(embedding)
		# print('213: {}'.format(embedding_new.requires_grad))

		# Extract the new word embedding by index
		word_embedding = embedding_new[word_idx, :, :]
		# print('217: {}'.format(word_embedding.requires_grad))
		# print(word_embedding)

		# Run fine-tuning MLP on new word embedding and get sense embedding
		sense_embedding = self._run_fine_tune_MLP(word_embedding, word_lemma, param = "word_sense")
		sense_embedding = sense_embedding.view(1, -1)

		# print('223: {}'.format(sense_embedding.requires_grad))
		# print(sense_embedding)
		return sense_embedding


	# 3 * 1024 -> 256 by dimension reduction
	def _tune_embeddings(self, embedding):
		return torch.tanh(self.dimension_reduction_MLP(embedding))
	
	def _run_fine_tune_MLP(self, word_embedding, word_lemma, param = "word_sense"):
		
		'''
		Runs MLP on the target word embedding
		'''
		for i, layer in enumerate(self.layers[param]):
			if i:
				word_embedding = layer(word_embedding)
				word_embedding = self.mlp_dropout(word_embedding)

		# print('\nWord lemma: {}\nWord sense embedding size: {}\nAll its senses: {}'.format(word_lemma, word_embedding.size(), self.all_senses[word_lemma]))
		return word_embedding
