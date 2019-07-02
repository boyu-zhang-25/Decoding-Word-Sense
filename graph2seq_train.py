
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

from graph_lstm import *
from decoder import *
from graph2seq_model import *

# get the decoder vocab
with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    print("Size of vocab: {}".format(vocab.idx))
    
# get the graph lstm synset vocab
with open('./data/synset_vocab.pkl', 'rb') as f:
    synset_vocab = pickle.load(f)
print("Size of synset vocab: {}".format(synset_vocab.idx))

# get the graph lstm synset vocab that only appears in the SemCor
with open('./data/synset_vocab_SemCor.pkl', 'rb') as f:
    synset_vocab_SemCor = pickle.load(f)
print("Size of synset vocab in SemCor: {}".format(synset_vocab_SemCor.idx))


# In[4]:


# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# data parallel for the decoder
# not working for the decoder here since we are using SGD per node
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")


# In[5]:


# some hyperparameters
max_seq_length = 20
decoder_hidden_size = 256
decoder_input_size = 512
mer_holo_depth = 5
hyper_hypon_depth = 5

# set the hyper-hypon and mer-holo graph lstms
hyper_hypon_graph = ChildSumGraphLSTM_WordNet(
    synset_vocab = synset_vocab, 
    relationship = 'hyper_hypon', 
    input_size = 256, 
    hidden_size = 64, 
    num_layers = 2, 
    bidirectional = True, 
    bias = True, 
    dropout = 0.2)

mer_holo_graph = ChildSumGraphLSTM_WordNet(
    synset_vocab = synset_vocab, 
    relationship = 'mer_holo',
    input_size = 256, 
    hidden_size = 64, 
    num_layers = 2, 
    bidirectional = True, 
    bias = True, 
    dropout = 0.2)

# decoder
# input size of decoder = 512
# output size of per graph lstm = 2 * 64 = 128
# concat graph embedding size = 256
# concat word embedding size (decoder_hidden_size) = 512 = input size of decoder
decoder = Decoder(
    vocab_size = vocab.idx, 
    max_seq_length = max_seq_length, 
    hidden_size = decoder_hidden_size, 
    input_size = decoder_input_size)

# the model instance
graph2seq_model = Graph2Seq_Model(
    hyper_hypon_graph,
    mer_holo_graph,
    hyper_hypon_depth,
    mer_holo_depth, 
    decoder, 
    vocab = vocab, 
    max_seq_length = max_seq_length, 
    decoder_hidden_size = decoder_hidden_size)

# randomly initialize the weights
def init_weights(m):
    for name, param in m.named_parameters():
        '''
        if param.requires_grad:
            print(name, param.shape)
        '''
        nn.init.uniform_(param.data, -0.08, 0.08)  
graph2seq_model.apply(init_weights)

# cuda
graph2seq_model.to(device)

# use pretrained checkpoint
pretrain = False
path = './models/graph2seq_best_model.pth'
if pretrain:
    graph2seq_model.load_state_dict(torch.load(path))


# In[6]:


# training hyperparameters
optimizer = optim.Adam(graph2seq_model.parameters())
PAD_IDX = vocab('<pad>')
print('PAD_IDX: {}'.format(PAD_IDX))
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX).to(device)


# In[7]:


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
  


# In[8]:


# the training function
# pretrain on the synsets appeared in the SemCor
small_size = 10
def train(model, optimizer, synset_vocab_SemCor, criterion, clip):
    
    model.train()
    epoch_loss = 0
    synset_num = synset_vocab_SemCor.idx
    
    # visualize results
    all_definitions = []
    all_sentence_result = []
    
    for idx in range(synset_vocab_SemCor.idx):
    # for idx in range(small_size):

        optimizer.zero_grad()
        
        # get the synset and definition
        synset = synset_vocab_SemCor.idx2word.get(idx).replace('__', '.')
        definition = wn.synset(synset).definition()
        # print(synset, definition)
        all_definitions.append(definition)
        
        # convert to def index list
        def_idx_list = def2idx(definition, model.max_length, vocab)

        # get the graph-decoder result
        # (self.max_length, batch_size, vocab_size) where batch_size == 1
        output, result = model(synset, def_idx_list, teacher_forcing_ratio = 0.4)
        all_sentence_result.append(result)

        # adjust dimension for loss calculation
        output = output.squeeze(1)
        target = torch.tensor(def_idx_list, dtype = torch.long).to(device)
        # print(target, definition)
        
        loss = criterion(output, target)
        # print(loss)
        loss.backward()

        # add clip for gradient boost
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()
                
    return epoch_loss / synset_num, all_sentence_result, all_definitions


# In[9]:


# time used by each epoch
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[10]:


# utility function
# arrange the result back to literal readable form
def arrange_result(all_sentence_result):
    arranged_all_sentence_result = []
    
    for result in all_sentence_result:
        arranged_results = ''
        
        for word_idx in result:
            w = ' '+ vocab.idx2word.get(int(word_idx))
            arranged_results += w
            
        arranged_all_sentence_result.append(arranged_results)
    return arranged_all_sentence_result


# In[11]:


# utility function
# write the results to the file, with the ground-truth
def write_result_to_file(arranged_all_sentence_result, all_definition):
    with open('result.txt', 'w') as f:
        for idx, arranged_results in enumerate(arranged_all_sentence_result):
            f.write("sentence {}\n".format(idx))
            f.write("%s\n" % all_definition[idx])
            f.write("%s\n" % arranged_results)
            f.write("\n")


# In[12]:


# train 
import time

N_EPOCHS = 40
CLIP = 1
best_train_loss = float('inf')
train_losses = []

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss, all_sentence_result, all_definitions = train(graph2seq_model, optimizer, synset_vocab_SemCor, criterion, CLIP)
    train_losses.append(train_loss)
        
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # visualize the results
    arranged_all_sentence_result = arrange_result(all_sentence_result)
            
    # save the best model based on the train set
    # since the nature of our pretrain without context, dev is useless
    if train_loss <= best_train_loss:

        best_train_loss = train_loss
        torch.save(graph2seq_model.state_dict(), './models/graph2seq_best_model.pth')
        
        # record the result
        write_result_to_file(arranged_all_sentence_result, all_definitions)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')


# In[13]:


# plot the learning curve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

with open('train_loss.tsv', mode = 'w') as loss_file:
    csv_writer = csv.writer(loss_file)
    csv_writer.writerow(train_losses)


# In[14]:


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

