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
				vocab,
				word_embed_size = 256,
				dropout = 0, 
				regularization = None,
				all_senses = None,
				device = device):
		super(Seq2Seq_Model, self).__init__()
		
		# taget word index and senses list
		self.all_senses = all_senses
		self.device = device

		# the word embedding size for the nn.Embedding
		# NOTICE: 1/2 of the decoder.embed_size because we will concat later
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

		self.encoder = encoder
		self.decoder = decoder
		self.max_length = decoder.max_seq_length
		self.vocab_size = decoder.vocab_size

		self.encoder = self.encoder.to(self.device)
		self.decoder = self.decoder.to(self.device)

		# word embedding for decoding sense
		self.pad_idx = vocab('<pad>')
		self.start_idx = vocab('<start>')
		self.end_idx = vocab('<end>')
		self.embed = nn.Embedding(decoder.vocab_size, self.word_embed_size, padding_idx = self.pad_idx)
		self.dropout = nn.Dropout(dropout)

	def forward(self, sentence, word_idx, definition, teacher_forcing_ratio = 0.4):
		
		'''
		teacher_forcing: the probability of using ground truth in decoding
		definnition: [1, self.max_length], the indices of each word in the true definition
		sentence: the given sentence in a list
		word_idx: the target word index in the sentence
		'''

		# SGD
		batch_size = 1
		
		# tensor to store decoder outputs
		outputs = torch.zeros(self.max_length, batch_size, self.vocab_size).to(self.device)

		# initialize the x_0 with <start>
		# (1, word_embed_size)
		lookup_tensor = torch.tensor([self.start_idx], dtype = torch.long)
		generated_embedding = self.dropout(self.embed(lookup_tensor))
		
		# sense embedding of the encode
		# (1, word_embed_size)
		encoder_embedding = self.encoder(sentence, word_idx)

		# concat of the encoder embedding and the generated word embedding
		# (1, decoder.embed_size)
		sense_embedding = torch.cat((encoder_embedding, generated_embedding), 1)

		# initialize h_0 and c_0 for the decoder LSTM_Cell
		hidden = torch.zeros(batch_size, self.decoder.hidden_size).to(self.device)
		cell = torch.zeros(batch_size, self.decoder.hidden_size).to(self.device)

		# explicitly iterate through the max_length to decode
		for t in range(0, self.max_length):
			
			# get embedding at each time step from the LSTM
			# (batch, vocab_size), (batch, decoder.hidden_size)
			output, hidden, cell = self.decoder(sense_embedding, hidden, cell)
			outputs[t] = output

			# get the max word index from the vocabulary
			generated_index = output.max(1)[1]

			# may use the correct word from the definition
			teacher_force = random.random() < teacher_forcing_ratio
			if teacher_force:
				word_index = definition[t]
			else:
				word_index = generated_index

			# get the new embedding
			# concat the encoder embedding to the generated embedding at each time step
			lookup_tensor = torch.tensor([word_index], dtype = torch.long)
			generated_embedding = self.dropout(self.embed(lookup_tensor))
			sense_embedding = torch.cat((encoder_embedding, generated_embedding), 1)

		return outputs
