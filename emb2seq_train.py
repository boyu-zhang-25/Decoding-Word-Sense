
# coding: utf-8

# In[1]:


import csv
import math
import string
import itertools
from io import open
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Iterable, defaultdict
import random
from nltk.tree import Tree
from nltk.corpus import semcor


# In[2]:


# set determinstic results
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# In[3]:


from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder()

from encoder import *
from decoder import *
from emb2seq_model import *

# get the decoder vocab
with open('./data/vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)
	print("Size of vocab: {}".format(vocab.idx))


# In[4]:


decoder = Decoder(vocab_size = vocab.idx)
encoder = Encoder(elmo_class = elmo)
emb2seq_model = Emb2Seq_Model(encoder, decoder, vocab = vocab)

# randomly initialize the weights
def init_weights(m):
	for name, param in m.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)       
emb2seq_model.apply(init_weights)


# In[5]:


# training hyperparameters
optimizer = optim.Adam(emb2seq_model.parameters())
PAD_IDX = vocab('<pad>')
print('PAD_IDX: {}'.format(PAD_IDX))
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)


# In[6]:


# utility function
# turn the given definition into its index list form
def def2idx(definition, max_length, vocab):
	
	# definition is given by the WN NLTK API in a string
	def_tokens = nltk.tokenize.word_tokenize(definition.lower())
	
	# limit the length if too long, trim
	if len(def_tokens) > (max_length - 2):
		def_tokens = def_tokens[0:(max_length - 2)]
		
		# add the start and end symbol
		def_tokens = ['<start>'] + def_tokens + ['<end>']
	
	# if the length is too short, pad
	elif len(def_tokens) < (max_length - 2):
		
		# add the start and end symbol
		def_tokens = ['<start>'] + def_tokens + ['<end>']
		
		pad = ['<pad>'] * (max_length - len(def_tokens))
		def_tokens = def_tokens + pad
		
	else:
		def_tokens = ['<start>'] + def_tokens + ['<end>']
			
	# get the index for each element in the token list
	def_idx_list = [vocab(token) for token in def_tokens]
	
	return def_idx_list
  


# In[11]:


small_train_size = 100
small_dev_size = 20

# the training function
def train(model, optimizer, criterion, clip):
	
	model.train()
	epoch_loss = 0
	
	# SGD
	# small test
	for idx in range(small_train_size):
				
		optimizer.zero_grad()
		
		# get the semcor tagged sentence
		sentence = semcor.sents()[idx]
		tagged_sent = semcor.tagged_sents(tag = 'sem')[idx]
		# print(idx)
		
		# get all-word definitions
		# [batch_size, self.max_length]
		definitions = []
		for chunk in tagged_sent:
			
			# if is tagged
			if isinstance(chunk, Tree):
				
				# if is an ambiguous word
				if isinstance(chunk.label(), nltk.corpus.reader.wordnet.Lemma):
					# print(chunk.label())
					synset = chunk.label().synset().name()
					definition = wn.synset(synset).definition()
					def_idx_list = def2idx(definition, model.max_length, vocab)
					definitions.append(def_idx_list)
		
		# get the encoder-decoder result
		# (self.max_length, batch_size, vocab_size)
		output, result = model(sentence, tagged_sent, definitions, teacher_forcing_ratio = 0.4)
		
		# adjust dimension for loss calculation
		output = output.permute(1, 2, 0)
		# print(output.shape)
		target = torch.tensor(definitions, dtype = torch.long)

		loss = criterion(output, target)
		loss.backward()
		
		# add clip for gradient boost
		torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		
		optimizer.step()
		epoch_loss += loss.item()
		
	return epoch_loss / small_test_size, result


# In[15]:


# evaluate the model
def evaluate(model, criterion):
	
	model.eval()
	epoch_loss = 0
	
	with torch.no_grad():
	
		for idx in range(small_dev_size):

			# get the semcor tagged sentence
			sentence = semcor.sents()[idx + small_train_size]
			tagged_sent = semcor.tagged_sents(tag = 'sem')[idx + small_train_size]
			print(idx)

			# get all-word definitions
			# [batch_size, self.max_length]
			definitions = []
			for idx, chunk in enumerate(tagged_sent):
				if isinstance(chunk, Tree):

					# only take in ambiguous words
					if isinstance(chunk.label(), nltk.corpus.reader.wordnet.Lemma):
						# print(chunk.label())
						synset = chunk.label().synset().name()
						definition = wn.synset(synset).definition()
						
						print(definition)
						def_idx_list = def2idx(definition, model.max_length, vocab)
						definitions.append(def_idx_list)

			# get the encoder-decoder result
			# (max_length, batch_size, vocab_size)
			# turn off teacher forcing
			output, result = model(sentence, tagged_sent, definitions, teacher_forcing_ratio = 0)

			# adjust dimension for loss calculation
			output = output.permute(1, 2, 0)
			# print(output.shape)
			target = torch.tensor(definitions, dtype = torch.long)

			loss = criterion(output, target)            
			epoch_loss += loss.item()
		
	return epoch_loss / small_dev_size, result


# In[16]:


# time used by each epoch
def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs


# In[17]:


# train and evaluate
import time

N_EPOCHS = 50
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
	
	start_time = time.time()
	
	train_loss, _ = train(emb2seq_model, optimizer, criterion, CLIP)
	valid_loss, result = evaluate(emb2seq_model, criterion)
	print(result)
	
	end_time = time.time()
	
	epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	
	# visualize the results
	all_results = []
	for n in range(len(result[0])):
		sense = ''
		for m in range(len(result)):
			w = ' ' + vocab.idx2word.get(int(result[m][n]))
			sense += w
	all_results.append(sense)

	with open('all_results.txt', 'w') as f:
		for sense in all_results:
			f.write("%s\n" % sense)
	
	# save the best model based on the dev set

	if valid_loss <= best_valid_loss:
		best_valid_loss = valid_loss
		torch.save(seq2seq_model.state_dict(), 'best_model.pth')
	
	print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
	print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
	print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

