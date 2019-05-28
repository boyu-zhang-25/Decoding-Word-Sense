import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Iterable, defaultdict
import itertools
from nltk.corpus import wordnet as wn
from encoder import *
from decoder import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))
print(torch.cuda.device_count())

class Model(nn.Module):

	def __init__(self, 
				optimizer_class = torch.optim.Adam,
				optim_wt_decay = 0.,
				epochs = 5,
				regularization = None,
				all_senses = None,
				all_supersenses = None, 
				elmo_class = None, # for sense vector in the model
				file_path = "",
				device = device,
				**kwargs):
		super(Model, self).__init__()

		## Training parameters
		self.epochs = epochs
		self.elmo_class = elmo_class

		## optimizer 
		self.optimizer = optimizer_class
		self.optim_wt_decay = optim_wt_decay
		
		# taget word index and senses list
		self.all_senses = all_senses

		self._init_kwargs = kwargs
		self.device = device

		# loss for the decoder
		# ignore the padding
		# TODO: add vocab
		self.loss = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
		
		'''
		if regularization == "l1":
			self.regularization = L1Loss()
		elif regularization == "smoothl1":
			self.regularization = SmoothL1Loss()
		else:
			self.regularization = None
		'''
		self.best_model_file =  file_path + "word_sense_model_.pth"		
		'''
		if self.regularization:
			self.regularization = self.regularization.to(self.device)
		'''

		self.encoder = Encoder(all_senses = self.all_senses, elmo_class = self.elmo_class, mlp_dropout = 0)

		# TODO: vocab
		self.decoder = Decoder(vocab_size = 0, max_seq_length = 15, teacher_forcing = 0.4)
		self.encoder = self.encoder.to(self.device)
		self.decoder = self.decoder.to(self.device)

	def forward(self):

	
