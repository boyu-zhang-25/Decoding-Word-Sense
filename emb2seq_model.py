import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Iterable, defaultdict
import itertools
import random
from encoder import *
from decoder import *
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
baseline model: encoder-decoder
feed the encoder result directly to the decoder
'''
class Emb2Seq_Model(nn.Module):

	def __init__(self,
				encoder,
				decoder,
				vocab,
				max_seq_length,
				decoder_hidden_size,
				word_idx_in_order,
				word_embed_size = 256,
				dropout = 0.2, 
				regularization = None,
				device = device):
		super(Emb2Seq_Model, self).__init__()
		self.device = device

		# the word embedding size for the nn.Embedding
		# NOTICE: 1/2 of the decoder.embed_size because we will concat later
		self.word_embed_size = word_embed_size

		# words in the decoder vocab in order
		# converted to the vocab idx for the transformer-xl
		self.word_idx_in_order = word_idx_in_order

		self.encoder = encoder
		self.decoder = decoder
		self.max_length = max_seq_length
		self.vocab_size = vocab.idx
		self.decoder_hidden_size = decoder_hidden_size

		# word embedding for decoding sense
		self.pad_idx = vocab('<pad>')
		self.start_idx = vocab('<start>')
		self.end_idx = vocab('<end>')
		self.embed = nn.Embedding(vocab.idx, self.word_embed_size, padding_idx = self.pad_idx)
		self.dropout = nn.Dropout(dropout)

	# given the generated sequence and memory from the previous step
	# get the log probability distribution over the vocab by the transfomer-xl model 
	def _get_trans_prob(self, trans_model, result, batch_size, mem):

		# get the idx (batch, seq_length) for the transformer
		# only after the first 3 time step
		context = self._get_trans_idx(result, batch_size)
		if mem == -1:
			# Predict all tokens
			predictions, mems = trans_model(context)
		else:
			# We can re-use the memory cells in a subsequent call to attend a longer context
			predictions, mems = trans_model(context, mems = mem)

		# get the log probability for predicted last token
		# for the subset vocab for our model
		our_prediction = torch.zeros(batch_size, len(self.word_idx_in_order)).to(torch.device('cuda:2'))
		for our_idx, xl_idx in enumerate(self.word_idx_in_order):
			our_prediction[:, our_idx] = predictions[:, -1, xl_idx]

		# (batch_size, vocab)
		return our_prediction, mems

	# helper method: get the idx form of the current batch
	# (batch, seq) for the transformer-xl
	def _get_trans_idx(self, result, batch_size):
		context = torch.zeros(batch_size, len(result), dtype = torch.long).to(torch.device('cuda:2'))
		for b in range(batch_size):
			for l in range(len(result)):
				context[b, l] = self.word_idx_in_order[result[l][b].item()]
		return context

	# perform all-word WSD on the SemCor dataset
	def forward(self, sentence, tagged_sent, definition, trans_model, teacher_forcing_ratio = 0.4):
		
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
		batch_size = len(tagged_sent)
		
		# tensor to store decoder outputs
		outputs = torch.zeros(self.max_length, batch_size, self.vocab_size).to(self.device)

		# initialize the x_0 with <start>
		# (batch_size, word_embed_size)
		lookup_tensor = torch.tensor([self.start_idx], dtype = torch.long).to(self.device)
		generated_embedding = self.dropout(self.embed(lookup_tensor)).repeat(batch_size, 1).to(self.device)
		# print(generated_embedding.shape)

		# concat of the encoder embedding and the generated word embedding
		# (batch_size, decoder.input_size)
		sense_embedding = torch.cat((encoder_embedding, generated_embedding), 1).to(self.device)

		# initialize h_0 and c_0 for the decoder LSTM_Cell
		hidden = torch.zeros(batch_size, self.decoder_hidden_size).to(self.device)
		cell = torch.zeros(batch_size, self.decoder_hidden_size).to(self.device)

		# visualize the result
		# store decoding context for the transformer-xl
		# 'result': list[tensor] of size (seq_length, batch) 
		result = []
		mem = -1

		# explicitly iterate through the max_length to decode
		for t in range(self.max_length):
			
			# get embedding at each time step from the LSTM
			# (batch, vocab_size), (batch, decoder.hidden_size)
			output, hidden, cell = self.decoder(sense_embedding, hidden, cell)
			# print('deocder out size in model: {}'.format(output.shape))
			# print(output.shape)
			outputs[t] = output

			# correct grammar for the final word choice
			# only after the first 3 time step
			if t > 3:

				# inference only, saving mem
				with torch.no_grad():

					trans_prob, mem = self._get_trans_prob(trans_model, result, batch_size, mem)

					# since trans-xl is on cuda 2, move it back to the default cuda
					trans_prob = trans_prob.to(device)

				# using convex combination with the transformer-xl
				alpha = 0.5
				output = alpha * output + (1 - alpha) * trans_prob

			# get the max word index from the vocabulary
			_, generated_index = torch.max(output, dim = 1)
			result.append(generated_index)

			# final word choices for all words in this sentence
			word_index = []

			# get the generated word index for each word in the batch
			# may use the correct word from the definition
			# store the context for the transformer-xl
			for batch in range(batch_size):
				teacher_force = random.random() < teacher_forcing_ratio
				if teacher_force:
					word_index.append(definition[batch][t])
				else:
					word_index.append(generated_index[batch])

			# get the new embedding
			# concat the encoder embedding to the generated embedding at each time step
			lookup_tensor = torch.tensor(word_index, dtype = torch.long).to(self.device)
			generated_embedding = self.dropout(self.embed(lookup_tensor)).to(self.device)
			# print(generated_embedding.shape)			
			sense_embedding = torch.cat((encoder_embedding, generated_embedding), 1).to(self.device)

		# print('model output size: {}'.format(outputs.shape))
		return outputs, result
