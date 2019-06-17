import torch
import torch.nn as nn
import numpy as np
import math

'''
a LSTM-based sequence decoder to decode the word sense in the given context
'''
class Decoder(nn.Module):

	def __init__(self,
				vocab_size,
				max_seq_length, # max length of definition generated is 18 with <start> and <end>
				hidden_size, # length of the generated word embedding 
				input_size = 512, # concat the sense embedding and the generated word embedding (2 * 256)
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

		"""Build the layers in the decoder."""
		super(Decoder, self).__init__()

		self.hidden_size = hidden_size
		self.device = device
		self.vocab_size = vocab_size
		self.input_size = input_size
		
		# the decoding LSTM
		self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size)
		
		# project the output from LSTM to vocabulary space
		self.linear = nn.Linear(self.hidden_size, self.vocab_size)

		# max length of sense used during decoding
		self.max_seq_length = max_seq_length 
		
	# this forward method only takes 1 time step
	# explicitly iterate through the max_length to decode
	# see the forward method of 'seq2seq_model' 
	def forward(self, sense_embedding, hidden, cell):
		'''
		the predicted word in the embedding:
		sense_embedding (batch_size, input_size): concat of the encoder embedding and the generated word embedding
		hidden, cell of shape (batch, hidden_size)
		'''

		# LSTM for 1 time step: generating the next embedding
		# (batch, hidden_size)
		(hidden, cell) = self.lstm_cell(sense_embedding, (hidden, cell))
		
		# project to the vocab
		# (batch_size, vocab_size)
		output = self.linear(hidden)
		# print('decoder out size: {}'.format(output.shape))
		return output, hidden, cell
