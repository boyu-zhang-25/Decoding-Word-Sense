import torch
import torch.nn as nn
import numpy as np
import math

'''
a LSTM-based sequence decoder to decode the word sense in the given context
'''
class Decoder(nn.Module):

	def __init__(self, 
				embed_size = 300, # the same as the output_size of the encoder
				hidden_size = 300, 
				vocab_size, # TODO: vocab
				num_layers = 1, # 1 layer LSTM for decoder
				max_seq_length = 15, # max length of definition generated is 15
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

		"""Build the layers in the decoder."""
		# TODO: add vocab

		super(Decoder, self).__init__()

		self.hidden_size = hidden_size
		self.device = device
		self.vocab_size = vocab_size
		
		# the decoding LSTM
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
		
		# project the output from LSTM to vocabulary space
		self.linear = nn.Linear(hidden_size, vocab_size)

		# max length of sense used during decoding
		self.max_seg_length = max_seq_length 
		
	def forward(self, sense_embedding, hidden, cell):
		'''
		the predicted word in the embedding:
		sense_embedding: (batch_size, embed_size)
		
		batch_size = 1 for SGD

		used to specify the state of the LSTM:
		hidden = [n layers * n directions, batch size, hid dim]
		cell = [n layers * n directions, batch size, hid dim]
		'''

		self.lstm.flatten_parameters()

		# (1, batch_size, embed_size)
		sense_embedding = sense_embedding.unsqueeze(0)

		# LSTM for 1 time step: generating the next embedding
		sense_embedding, (hidden, cell) = self.lstm(sense_embedding, (hidden, cell))
		
		# output: (batch_size, vocab_size)
		output = self.linear(sense_embedding.squeeze(0))

		return output, hidden, cell
