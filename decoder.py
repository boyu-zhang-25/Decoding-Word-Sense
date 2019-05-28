import torch
import torch.nn as nn
import numpy as np
import math
from nltk.corpus import wordnet as wn

'''
a LSTM-based sequence decoder to decode the word sense in the given context
'''
class Decoder(nn.Module):

	def __init__(self, 
				embed_size = 256, 
				hidden_size = 300, # the same as the output_size of the encoder
				vocab_size, # TODO: vocab
				num_layers = 1, # 1 layer LSTM for decoder
				dropout = 0, 
				max_seq_length = 15, # max length of definition generated is 15
				teacher_forcing, # the probability of using ground truth in decoding
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

		"""Build the layers in the decoder."""
		# TODO: add vocab

		super(Decoder, self).__init__()

		self.dropout = dropout
		self.hidden_size = hidden_size
		self.device = device

		# word embedding for sense
		self.embed = nn.Embedding(vocab_size, embed_size)
		
		# the decoding LSTM
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
		
		# project the output from LSTM to vocabulary space
		self.linear = nn.Linear(hidden_size, vocab_size)

		# max length of sense used during decoding
		self.max_seg_length = max_seq_length 

		# the probability of using the ground truth during decoding
		self.teacher_forcing = teacher_forcing
		
	def forward(self, word_index, hidden, cell):
		'''
		the predicted word in the embedding:
		word_index: (batch_size)

		used to specify the state of the LSTM:
		hidden = [n layers, batch size, hid dim]
		cell = [n layers, batch size, hid dim]
		'''

		# extract the embedding of the predicted word
		word_index = word_index.unsqueeze(0)
		# word: (1, batch_size, embed_size)
		word = self.dropout(self.embed(word_index))

		# word_embed: (1, batch_size, hidden_size)
		# preserve the new LSTM weights
		self.lstm.flatten_parameters()
		word_embed, (hidden, cell) = self.lstm(word, (hidden, cell))
		
		# output: (batch_size, vocab_size)
		output = self.linear(word_embed.squeeze(0))

		return output, hidden, cell
	
	def sample(self, embedding, states = None):
		'''
		Generate sense definition for the given word 
		from the given sense_embedding with a word-by-word scheme.
		'''
 
 		sampled_ids = []
		states = (torch.randn(1, 1, self.hidden_size).to(self.device), torch.randn(1, 1 self.hidden_size).to(self.device))

		# size: (1, 1, encoder embedding size)
		inputs = embedding

		for i in range(self.max_seg_length):

			# hiddens: (sequence length, 1, hidden_size)
			hiddens, states = self.lstm(inputs, states)

			# output: (sequence length, 1, vocab_size)
			output = self.linear(hiddens.squeeze(1))

			# choose the max as predicted word from the vocab
			_, predicted = output.max(1)
			sampled_ids.append(predicted)
			inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
			inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
		sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
		return sampled_ids