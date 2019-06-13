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
class Emb2Seq_Model(nn.Module):

	def __init__(self,
				encoder,
				decoder,
				vocab,
				word_embed_size = 256,
				dropout = 0, 
				regularization = None,
				device = device):
		super(Emb2Seq_Model, self).__init__()
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

	# perform all-word WSD on the SemCor dataset
	def forward(self, sentence, tagged_sent, definition, teacher_forcing_ratio = 0.4):
		
		'''
		teacher_forcing: the probability of using ground truth in decoding

		definition: [seq_length, self.max_length]
		the matrix with row as seq and column as words: indices of each word in the true definition

		sentence: the given plain sentence in a list
		tagged_sent: the target sentence (list) with only tagged words from the SemCor
		'''
		
		# sense embedding from the encoder
		# (seq_length, 256): batch is the seq_length for all-word WSD
		encoder_embedding = self.encoder(sentence, tagged_sent)
		# print(encoder_embedding.shape)

		# treating one sentence as a batch for all-word WSD
		# each word is an example for the decoder
		batch_size = encoder_embedding.shape[0]
		
		# tensor to store decoder outputs
		outputs = torch.zeros(self.max_length, batch_size, self.vocab_size).to(self.device)

		# initialize the x_0 with <start>
		# (batch_size, word_embed_size)
		lookup_tensor = torch.tensor([self.start_idx], dtype = torch.long)
		generated_embedding = self.dropout(self.embed(lookup_tensor)).repeat(batch_size, 1)
		# print(generated_embedding.shape)

		# concat of the encoder embedding and the generated word embedding
		# (batch_size, decoder.embed_size)
		sense_embedding = torch.cat((encoder_embedding, generated_embedding), 1)

		# initialize h_0 and c_0 for the decoder LSTM_Cell
		hidden = torch.zeros(batch_size, self.decoder.hidden_size).to(self.device)
		cell = torch.zeros(batch_size, self.decoder.hidden_size).to(self.device)

		# visualize the result
		result = []

		# explicitly iterate through the max_length to decode
		for t in range(self.max_length):
			
			# get embedding at each time step from the LSTM
			# (batch, vocab_size), (batch, decoder.hidden_size)
			output, hidden, cell = self.decoder(sense_embedding, hidden, cell)
			# print(output.shape)
			outputs[t] = output

			# get the max word index from the vocabulary
			_, generated_index = torch.max(output, dim = 1)
			result.append(generated_index)

			# final word choices for all words in this sentence
			word_index = []

			# get the generated word index for each word in the batch
			# may use the correct word from the definition
			# print('batch: {}'.format(batch_size))
			for batch in range(batch_size):
				teacher_force = random.random() < teacher_forcing_ratio
				if teacher_force:
					word_index.append(definition[batch][t])
				else:
					word_index.append(generated_index[batch])

			# get the new embedding
			# concat the encoder embedding to the generated embedding at each time step
			lookup_tensor = torch.tensor(word_index, dtype = torch.long)
			generated_embedding = self.dropout(self.embed(lookup_tensor))
			# print(generated_embedding.shape)			
			sense_embedding = torch.cat((encoder_embedding, generated_embedding), 1)

		return outputs, result
