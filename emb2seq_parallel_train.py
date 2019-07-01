
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


# In[2]:


# set determinstic results
'''
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
'''


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

# some hyperparameters
max_seq_length = 17
decoder_hidden_size = 256

decoder = Decoder(
	vocab_size = vocab.idx, 
	max_seq_length = max_seq_length, 
	hidden_size = decoder_hidden_size)

encoder = Encoder(elmo_class = elmo)


# In[5]:


# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# the auxiliary transformer-xl for the decoder grammar
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

# Load pre-trained model tokenizer (vocabulary from wikitext 103)
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
# Load pre-trained model (weights)
trans_model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
trans_model.eval()

# put on another GPU for memory issues
trans_model.to(device = torch.device(device))

word_idx_in_order = [tokenizer.convert_tokens_to_ids([vocab.idx2word.get(idx)])[0] for idx in range(vocab.idx)]

# create the model instance
emb2seq_model = Emb2Seq_Model(
	encoder, 
	decoder, 
	vocab = vocab, 
	max_seq_length = max_seq_length, 
	decoder_hidden_size = decoder_hidden_size,
	word_idx_in_order = word_idx_in_order)

emb2seq_model.to(device)
optimizer = optim.Adam(emb2seq_model.parameters())

# randomly initialize the weights
def init_weights(m):
	for name, param in m.named_parameters():
		'''
		if param.requires_grad:
			print(name, param.shape)
		'''
		nn.init.uniform_(param.data, -0.08, 0.08)   
emb2seq_model.apply(init_weights)
# print(emb2seq_model.decoder.state_dict())

# use pretrained decoder
pretrain = False
pretrain_path = './models/decoder.pth'
if pretrain:
	emb2seq_model.decoder.load_state_dict(torch.load(pretrain_path))
	# print(emb2seq_model.decoder.state_dict())

# data parallel for the decoder
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	emb2seq_model.decoder = nn.DataParallel(emb2seq_model.decoder)

# ignore idx on the padding
PAD_IDX = vocab('<pad>')
print('PAD_IDX: {}'.format(PAD_IDX))
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX).to(device)


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
  


# In[7]:


# utility function
# get the literal and indices of the definition of the tagged word
# SemCor for train
def get_SemCor_def(instance):
	key = ''
	target_file = open("../WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt", "r")
	for line in target_file:
		if line.startswith(instance.get('id')):
			key = line.replace('\n', '').split(' ')[-1]
			
	# literal definition from the WN
	definition = wn.lemma_from_key(key).synset().definition()
	return definition

# SemEval 2007 for dev
def get_SemEval_def(instance):
	key = ''
	target_file = open("../WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt", "r")
	for line in target_file:
		if line.startswith(instance.get('id')):
			key = line.replace('\n', '').split(' ')[-1]
			
	# literal definition from the WN
	definition = wn.lemma_from_key(key).synset().definition()
	return definition


# In[8]:


# parse the SemCor training data
import xml.etree.ElementTree as ET
semcor_tree = ET.parse('../WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml')
semcor_corpus = semcor_tree.getroot()

# parse the SemEval dev data
semeval_tree = ET.parse('../WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.data.xml')
semeval_corpus = semeval_tree.getroot()

# small train and test sets
# small_train_size = 1
# small_dev_size = 1


# In[9]:


# the training function
def train(model, optimizer, corpus, criterion, clip):
	
	model.train()
	epoch_loss = 0
	sentence_num = 0
	i = 0

	for sub_corpus in corpus:
	
		for sent in sub_corpus:

			optimizer.zero_grad()
			
			# get the plain text sentence
			sentence = [word.text for word in sent]
			
			# get the tagged ambiguous words
			tagged_sent = [instance for instance in sent if instance.tag == 'instance']
			# print(sentence)
			# print(tagged_sent)
			
			# only use sentence with at least one tagged word
			if len(tagged_sent) > 0:
				
				sentence_num += 1
				
				# get all-word definitions, batch_size is the sentence length
				# [batch_size, self.max_length]
				definitions = []
				for instance in tagged_sent:
					
					# get the sense from the target file with ID
					definition = get_SemCor_def(instance)
					def_idx_list = def2idx(definition, model.max_length, vocab)
					definitions.append(def_idx_list)

				# get the encoder-decoder result
				# (self.max_length, batch_size, vocab_size)
				output, _ = model(
								sentence, 
								tagged_sent, 
								definitions, 
								trans_model, 
								teacher_forcing_ratio = 0.4)
				
				# adjust dimension for loss calculation
				# (self.max_length * batch_size, vocab_size)
				output = output.view(-1, output.shape[-1])
				target = torch.tensor(definitions, dtype = torch.long).to(device)

				# (self.max_length * batch_size)
				target = torch.transpose(target, 0, 1).contiguous().view(-1)
				'''
				output = output.permute(1, 2, 0)
				target = torch.tensor(definitions, dtype = torch.long).to(device)
				'''
				loss = criterion(output, target)
				loss.backward()

				# add clip for gradient boost
				# torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

				optimizer.step()
				epoch_loss += loss.item()

		# keep track of progress
		i += 1
		if i % 10 == 0:
			print("[{}/{}] subcorpus done.".format(i, len(corpus)))
				
	return epoch_loss / sentence_num


# In[10]:


# evaluate the model
def evaluate(model, corpus, criterion):
	
	model.eval()
	epoch_loss = 0
	sentence_num = 0

	# result from all dev sentences, both idx form and literal form
	all_definitions = []
	all_sentence_result = []
	
	with torch.no_grad():
	
		for sub_corpus in corpus:
	
			for sent in sub_corpus:
				
				sentence = [word.text for word in sent]
				
				# get the tagged ambiguous words
				tagged_sent = [instance for instance in sent if instance.tag == 'instance']
				# print(sentence)
				# print(tagged_sent)

				# only use sentence with at least one tagged word
				if len(tagged_sent) > 0:
					sentence_num += 1

					# get all-word definitions, batch_size is the sentence length
					# [batch_size, self.max_length]
					definitions = []
					literal_def = []
					for instance in tagged_sent:

						# get the sense from the target file with ID
						definition = get_SemEval_def(instance)
						def_idx_list = def2idx(definition, model.max_length, vocab)
						definitions.append(def_idx_list)
						literal_def.append(definition)

					# get the encoder-decoder result
					# (self.max_length, batch_size, vocab_size)
					output, result = model(
										sentence, 
										tagged_sent, 
										definitions, 
										trans_model, 
										teacher_forcing_ratio = 0)

					all_sentence_result.append(result)
					all_definitions.append(literal_def)
					
					# adjust dimension for loss calculation
					# (self.max_length * batch_size, vocab_size)
					output = output.view(-1, output.shape[-1])
					target = torch.tensor(definitions, dtype = torch.long).to(device)
					# (self.max_length * batch_size)
					target = torch.transpose(target, 0, 1).contiguous().view(-1)
					'''
					output = output.permute(1, 2, 0)
					target = torch.tensor(definitions, dtype = torch.long).to(device)
					'''
					loss = criterion(output, target)        
					epoch_loss += loss.item()
					
	return epoch_loss / sentence_num, all_sentence_result, all_definitions


# In[11]:


# time used by each epoch
def epoch_time(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time / 60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins, elapsed_secs


# In[12]:


# utility function
# arrange the result back to literal readable form
def arrange_result(all_sentence_result):
	arranged_all_sentence_result = []
	for result in all_sentence_result:
		arranged_results = []
		for n in range(len(result[0])):
			sense = ''
			for m in range(len(result)):
				w = ' '+ vocab.idx2word.get(int(result[m][n]))
				sense += w
			arranged_results.append(sense)
		arranged_all_sentence_result.append(arranged_results)
	return arranged_all_sentence_result


# In[13]:


# utility function
# write the results to the file, with the ground-truth
def write_result_to_file(arranged_all_sentence_result, all_definition):
	with open('emb2seq_result.txt', 'w') as f:
		for idx, arranged_results in enumerate(arranged_all_sentence_result):
			f.write("sentence {}\n".format(idx))

			for literal_def in all_definitions[idx]:
				f.write("%s\n" % literal_def)

			for item in arranged_results:
				f.write("%s\n" % item)
			f.write("\n")


# In[14]:


# train and evaluate
import time

N_EPOCHS = 40
CLIP = 1
best_valid_loss = float('inf')
train_losses = []
dev_losses = []

for epoch in range(N_EPOCHS):
	
	start_time = time.time()
	
	train_loss = train(emb2seq_model, optimizer, semcor_corpus, criterion, CLIP)
	train_losses.append(train_loss)
	
	valid_loss, all_sentence_result, all_definitions = evaluate(emb2seq_model, semeval_corpus, criterion)
	dev_losses.append(valid_loss)
		
	end_time = time.time()
	epoch_mins, epoch_secs = epoch_time(start_time, end_time)
	
	# visualize the results
	arranged_all_sentence_result = arrange_result(all_sentence_result)
			
	# save the best model based on the dev set
	if valid_loss <= best_valid_loss:

		best_valid_loss = valid_loss
		torch.save(emb2seq_model.state_dict(), './models/emb2seq_best_model.pth')
		
		# record the result
		write_result_to_file(arranged_all_sentence_result, all_definitions)
	
	print('Epoch: {:02} | Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_secs))
	print('\tTrain Loss: {:.3f} | Train PPL: {:7.3f}'.format(train_loss, math.exp(train_loss)))
	print('\t Val. Loss: {:.3f} |  Val. PPL: {:7.3f}'.format(valid_loss, math.exp(valid_loss)))


# In[15]:


# plot the learning curve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

with open('train_loss.tsv', mode = 'w') as loss_file:
	csv_writer = csv.writer(loss_file)
	csv_writer.writerow(train_losses)

with open('dev_loss.tsv', mode = 'w') as loss_file: 
	csv_writer = csv.writer(loss_file)
	csv_writer.writerow(dev_losses)


# In[16]:


plt.figure(1)
# rc('text', usetex = True)
rc('font', family='serif')
plt.grid(True, ls = '-.',alpha = 0.4)
plt.plot(train_losses, ms = 4, marker = 's', label = "Train Loss")
plt.legend(loc = "best")
title = "CrossEntropy Loss"
plt.title(title)
plt.ylabel('Loss')
plt.xlabel('Number of Iteration')
plt.tight_layout()
plt.savefig('train_loss.png')


# In[17]:


plt.figure(2)
# rc('text', usetex = True)
rc('font', family='serif')
plt.grid(True, ls = '-.',alpha = 0.4)
plt.plot(dev_losses, ms = 4, marker = 'o', label = "Dev Loss")
plt.legend(loc = "best")
title = "CrossEntropy Loss"
plt.title(title)
plt.ylabel('Loss')
plt.xlabel('Number of Iteration')
plt.tight_layout()
plt.savefig('dev_loss.png')

