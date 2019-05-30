import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Iterable, defaultdict
import itertools
import random
from encoder import *
from decoder import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))
print(torch.cuda.device_count())

'''
baseline model: encoder-decoder
feed the encoder result directly to the decoder
'''
class Seq2Seq_Model(nn.Module):

	def __init__(self,
				encoder,
				decoder,
				dropout = 0, 
				regularization = None,
				all_senses = None,
				device = device):
		super(Seq2Seq_Model, self).__init__()
		
		# taget word index and senses list
		self.all_senses = all_senses
		self.device = device
		
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

		self.encoder = encoder
		# TODO: vocab size
		self.decoder = decoder

		self.encoder = self.encoder.to(self.device)
		self.decoder = self.decoder.to(self.device)

		# word embedding for decoding sense
		self.embed = nn.Embedding(self.decoder.vocab_size, self.decoder.embed_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, sentence, word_idx, definition, teacher_forcing_ratio = 0.4):
		
		'''
		teacher_forcing: the probability of using ground truth in decoding
		definnition: [1, max_len], the indices of each word in the true definition
		sentence: the given sentence in a list
		word_idx: the target word index in the sentence
		'''

		# TODO: pre-process the definition

		# max definition and vocab length
		max_len = self.decoder.max_seq_length
		vocab_size = self.decoder.vocab_size

		# SGD
		batch_size = 1
		
		# tensor to store decoder outputs
		outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)
		
		# sense embedding produced by the encoder as x0 for the decoder
		# (1, 300)
		sense_embedding = self.encoder(sentence, word_idx)
		
		# first input to the decoder is the <sos> tokens
		word_index = definition[0]
		
		# initialize h_0 and c_0
		hidden = torch.zeros(1, batch_size, self.decoder.hidden_size).to(self.device)
		cell = torch.zeros(1, batch_size, self.decoder.hidden_size).to(self.device)

		# explicitly iterate through the max_length to decode
		for t in range(1, max_len):
			
			# get embedding at each time step from the LSTM
			# (batch, vocab_size)
			output, hidden, cell = self.decoder(sense_embedding, hidden, cell)
			outputs[t] = output

			# get the max word index from the vocabulary
			generated_index = output.max(1)[1]

			# may use the correct word from the definition
			teacher_force = random.random() < teacher_forcing_ratio
			if teacher_force and len(definition) > t:
				word_index = definition[t]
			else:
				word_index = generated_index

			# get the new embedding as the input of the next time step
			sense_embedding = self.dropout(self.embedding(word_index))
		
		return outputs
