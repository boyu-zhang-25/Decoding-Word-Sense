import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Iterable, defaultdict
import itertools
import nltk
from nltk.corpus import wordnet as wn

'''
the encoder to encode the target word sense in the given context
given a sequence (sentence)
generate the sense embedding of the word
'''
class Encoder(nn.Module):
	def __init__(self, 
				output_size = 256, # output size of each sense embedding [256, 1]
				embedding_size = 1024, # ELMo embedding size
				elmo_class = None,
				tuned_embed_size = 512,
				mlp_dropout = 0.1,
				lstm_hidden_size = 256, # bi-directional, so final size is 512
				MLP_size = 300, # 1 hidden layer for fine-tuning sense vector
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
		super(Encoder, self).__init__()
		
		# all senses for all words
		# useful for all purposes
		self.elmo_class = elmo_class
		self.device = device

		# for dimension reduction 
		self.tuned_embed_size = tuned_embed_size 
		self.embedding_size = embedding_size

		# sizes of the fine-tuning MLP and LSTM
		self.MLP_size = MLP_size
		self.output_size = output_size
		self.lstm_hidden_size = lstm_hidden_size
		self.mlp_dropout = mlp_dropout

		# dimension reduction for elmo
		# 3 * 1024 ELMo -> 512
		self.dimension_reduction = nn.Linear(self.embedding_size * 3, self.tuned_embed_size)

		# construct a LSTM on top of ELMo
		self.lstm = nn.LSTM(
						self.tuned_embed_size, 
						self.lstm_hidden_size, 
						num_layers = 2, 
						bidirectional = True, 
						dropout = 0.2)

		# build a 1-hidden-layer MLP on top of LSTM for fine-tuning
		self.mlp = nn.Sequential(
					nn.Linear(self.lstm_hidden_size * 2, self.MLP_size), 
					nn.ReLU(),
					nn.Dropout(p = self.mlp_dropout), 
					nn.Linear(self.MLP_size, self.output_size), 
					nn.Dropout(p = self.mlp_dropout))
	
	# for the input sentence (already tokenized and pooled phrases from SemCor)
	# get the ELMo embeddings of the given sentence		
	def _get_embedding(self, sentence):
		'''
			@param: a list of words (a sentence)

			@return: 
			ELMo embeddins of the sentence (for all three ELMo layers each 1024 dim)
			concatenates them to make a 3072 dim embedding vector
			reduces the dimension to a lower number (512)
		''' 

		# get ELMo embedding of the sentence
		# [3 (layers), sentence length, 1024 (word vector length)]
		embedding = torch.from_numpy(self.elmo_class.embed_sentence(sentence))
		embedding = embedding.to(self.device)

		# [sentence_length, 3 (layers), 1024 (word vector length)]
		embedding = embedding.permute(1, 0, 2)

		# concatenate 3 layers and reshape
		# [sentence_length, batch_size, 3 * 1024]
		batch_size = 1
		sentence_length = embedding.size()[0]
		embedding = embedding.contiguous().view(sentence_length, batch_size, -1)
		
		# [sentence_length, batch_size, 512]
		embedding = torch.tanh(self.dimension_reduction(embedding))

		return embedding


	# forward propagation selected sentence and definitions
	# all-word WSD from the SemCor dataset
	# tagged_sent is the list of XML elements for each word or phrase
	def forward(self, sentence, tagged_sent):
		
		# get the dimension-reduced ELMo embedding
		# [sentence_length, batch_size, 512]
		embedding = self._get_embedding(sentence)

		# Run a Bi-LSTM and get the sense embedding
		# (seq_len, batch, num_directions * hidden_size): batch = 1 here
		self.lstm.flatten_parameters()
		embedding_new, (hn, cn) = self.lstm(embedding)

		# Extract the new word embedding for all tagged words
		# (new_seq, num_directions * hidden_size)
		processed_embedding = self._process_embedding(sentence, embedding_new, tagged_sent)

		# Run fine-tuning MLP on new word embedding and get sense embedding
		# (new_seq, 256): new seq length is the number of tagged words/phrases
		sense_embedding = self.mlp(processed_embedding)
		# print(sense_embedding.shape)
		return sense_embedding

	# average pool the phrases and remove untagged words
	# deal with partially labeled or phrase-labeled sentences
	# tagged_sent is the list of tagged words from the SemCor
	def _process_embedding(self, sentence, embedding, tagged_sent):

		# only get those tagged phrases and words 
		new_embedding = [embedding[sentence.index(instance.text), :, :] for instance in tagged_sent]

		# concate the all-word embeddings
		# (new_seq, num_directions * hidden_size)
		result = torch.cat(new_embedding, dim = 0).to(self.device)
		return result
