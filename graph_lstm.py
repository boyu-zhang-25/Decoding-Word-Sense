import sys
import torch
import pdb
from abc import ABCMeta, abstractmethod
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.rnn import RNNBase
from torch.nn import Embedding
import time
import pickle
if sys.version_info.major == 3:
	from functools import lru_cache
else:
	from functools32 import lru_cache

from nltk.corpus import wordnet as wn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# test graph print
old_depth = 2

class ChildSumGraphLSTM(RNNBase):
	"""A bidirectional extension of child-sum tree LSTMs that work on graphs
	For each node, the embedding is calculated recursively by LSTM 
	The recursion will go through all connected nodes in the graph.
	Each node has its own embedding store in the synset embeddings
	It supports bidirection and stacked layers.

	This class cannot be instantiated directly. 
	Instead, the following subclasses runs on the WordNet:
	  - ChildSumGraphLSTM_WordNet
	  it supports two relationships (relationships) from the WordNet:
	  - hypernyms and hyponyms
	  - meronyms and holonyms
	"""

	__metaclass__ = ABCMeta

	def __init__(self, synset_vocab, relationship, *args, **kwargs):
		super(ChildSumGraphLSTM, self).__init__('LSTM', *args, **kwargs)

		# lru_cache is normally used as a decorator, but that usage
		# leads to a global cache, where we need an instance specific
		# cache
		self._get_parameters = lru_cache()(self._get_parameters)

		# all the WordNet synset embeddings and their indices
		self.synset_vocab = synset_vocab
		self.embedding = Embedding(synset_vocab.idx, self.input_size)

		# currently only supports (hypernyms, hyponyms) and (meronyms, holonyms)
		if relationship not in ['hyper_hypon', 'mer_holo']:
			print('Lexical Relationship Not Supported!')
			raise NotImplementedError
		else:
			self.relationship = relationship


	@staticmethod
	def nonlinearity(x):
		return torch.tanh(x)

	'''
	given a synset
	use all its hypers and hypons to generate the new node embedding
	recursively over the whole graph
	'''
	def forward(self, synset, depth):
		"""
		Parameters
		----------

		synset: the name of a 'nltk.corpus.reader.wordnet.Synset'
			the current synset (node)
			must be coverted to string by wn.synset().name()
			and convert '.' to '__' due to hashing

		depth: the max depth the recursion will go through the WordNet
			use -1 for a complete resursion over the graph (may hit the Python max recursion depth error)

		Due to the nltk.corpus.wordnet package support,
		user does not have to provide the graph to the forward function

		Returns
		-------
		hidden_all: list of torch.Tensor
			the updated hidden states of all connected nodes along the resursion process
		hidden_final, cell_final: torch.Tensor
			the final hidden state and cell state of the target synset node.
		"""

		# used to store all updated embeddings
		# of all the synsets that is connected and updated in this trun
		# {layers: {'up': {synset: [embedding tensor]}, 'down': {synset: [embedding tensor]}}}
		self.hidden_state = {}
		self.cell_state = {}

		for layer in range(self.num_layers):

			# hyper == 'up'
			# hypon == 'down'
			self.hidden_state[layer] = {'up': {}, 'down': {}}
			self.cell_state[layer] = {'up': {}, 'down': {}}

			# get the new node embedding by all its hypers and hypons
			# start with hyper
			self._upward_downward(layer, 'up', synset, depth, {'up': False, 'down': False})

		# convert '__' to '.' due to hashing
		synset = synset.replace('.', '__')

		# the hidden states of all connected hyper and hypons during the recursion
		# the final hidden state and cell state of the current synset
		hidden_up = self.hidden_state[self.num_layers - 1]['up']
		hidden_down = self.hidden_state[self.num_layers - 1]['down']

		# return all the hidden states tagged with the synset
		hidden_all = dict()
		# print(len(self.hidden_state[self.num_layers - 1]['up'].keys()))
		# print(len(self.hidden_state[self.num_layers - 1]['down'].keys()))

		for key in self.hidden_state[self.num_layers - 1]['up'].keys():
			hidden_all.update({key: torch.cat([hidden_up[key], hidden_down[key]])})

		hidden_final = torch.cat([hidden_up[synset], hidden_down[synset]])
		cell_final = torch.cat([self.cell_state[self.num_layers - 1]['up'][synset], self.cell_state[self.num_layers - 1]['down'][synset]])

		''''
		support mini-batch? 
		if self._has_batch_dimension:
			if self.batch_first:
				return hidden_all[None, :, :], hidden_final[None, :]
			else:
				return hidden_all[:, None, :], hidden_final[None, :]
		else:
			return hidden_all, hidden_final
		'''

		# (the standard output size of LSTM)
		return hidden_all, (hidden_final, cell_final)

	'''
	given the current node (synset, in '__')
	find all its hyper/hypon embeddings recursively (in _construct_previous)
	calculate the new embedding by the LSTM gates
	'''
	def _upward_downward(self, layer, direction, synset, depth, explored):

		# convert '__' to '.' due to hashing
		synset = synset.replace('.', '__')

		# 'explored' dictionary is used to monitor priority of the node
		# since the same node may by visited twice
		# allow over-writing on specific direction
		explored[direction] = True

		# check to see whether this node has been computed on this
		# layer in this direction, if so short circuit the rest of
		'''
		if synset in self.hidden_state[layer][direction]:

			# print('{}short-circuit synset: {}; direction: {}\n'.format('    ' * (old_depth - depth), synset, direction))
			h_t = self.hidden_state[layer][direction][synset]
			c_t = self.cell_state[layer][direction][synset]

			return h_t, c_t
		'''

		# get the current node x_t from the embedding
		x_t = self._construct_x_t(layer, synset)

		# construct the hyper and hypon embedding recursively
		# keep track of the depth of the recursion
		# h_prev, c_prev: (hidden_size, num_hyper/num_hypon)
		# convert '__' to '.' due to hashing
		synset = synset.replace('__', '.')
		oidx, (h_prev, c_prev) = self._construct_previous(layer, direction, synset, depth - 1, explored)

		# broadcasting and cast the tensor size 
		# to calculate all the LSTM gates
		if self.bias:
			Wih, Whh, bih, bhh = self._get_parameters(layer, direction)

			fcio_t_raw = torch.matmul(Whh, h_prev) +\
				torch.matmul(Wih, x_t[:, None]) +\
				bhh[:, None] + bih[:, None]

		else:
			Wih, Whh = self._get_parameters(layer, direction)

			fcio_t_raw = torch.matmul(Whh, h_prev) +\
				torch.matmul(Wih, x_t[:, None])

		# split for each LSTM gate
		f_t_raw, c_hat_t_raw, i_t_raw, o_t_raw = torch.split(fcio_t_raw,
															 self.hidden_size,
															 dim = 0)
		# hidden and cell states calculation
		# summing over the gated_children for new h and c
		f_t = torch.sigmoid(f_t_raw)

		gated_children = torch.mul(f_t, c_prev)
		gated_children = torch.sum(gated_children, 1, keepdim = False)

		c_hat_t_raw = torch.sum(c_hat_t_raw, 1, keepdim = False)
		i_t_raw = torch.sum(i_t_raw, 1, keepdim = False)
		o_t_raw = torch.sum(o_t_raw, 1, keepdim = False)

		c_hat_t = self.__class__.nonlinearity(c_hat_t_raw)
		i_t = torch.sigmoid(i_t_raw)
		o_t = torch.sigmoid(o_t_raw)

		c_t = gated_children + torch.mul(i_t, c_hat_t)
		h_t = torch.mul(o_t, self.__class__.nonlinearity(c_t))

		# may add dropout
		if self.dropout:
			dropout = Dropout(p = self.dropout)
			h_t = dropout(h_t)
			c_t = dropout(c_t)

		# store h and c for the new synset embeddings
		# allow overwrite by parents
		synset = synset.replace('.', '__')
		self.hidden_state[layer][direction][synset] = h_t
		self.cell_state[layer][direction][synset] = c_t

		# if bidirectional, get the embeddings of the opposite direction
		# otherwise, stop the recursion
		if explored['up'] == True and explored['down'] == False:
			self._upward_downward(layer, 'down', synset, depth, explored)
		elif explored['up'] == False and explored['down'] == True:
			self._upward_downward(layer, 'up', synset, depth, explored)
		elif explored['up'] == False and explored['up'] == False:
			print('both direction not explored')
			raise NotImplementedError

		# (hidden_size)
		return h_t, c_t

	# get the params of the LSTM
	# default params by the Pytorch source code of the RNNBase
	def _get_parameters(self, layer, direction):
		dirtag = '' if direction == 'up' else '_reverse'

		Wihattrname = 'weight_ih_l{}{}'.format(str(layer), dirtag)
		Whhattrname = 'weight_hh_l{}{}'.format(str(layer), dirtag)

		Wih, Whh = getattr(self, Wihattrname), getattr(self, Whhattrname)

		if self.bias:
			bhhattrname = 'bias_hh_l{}{}'.format(str(layer), dirtag)
			bihattrname = 'bias_ih_l{}{}'.format(str(layer), dirtag)

			bih, bhh = getattr(self, bihattrname), getattr(self, bhhattrname)

			return Wih, Whh, bih, bhh

		else:
			return Wih, Whh

	@abstractmethod
	def _construct_x_t(self, layer, synset):
		raise NotImplementedError

	'''
	given the current synset (node, in '.') and the direction
	recursively find all its hyper or hypon embeddings
	limited to the max recursion depth
	'''
	def _construct_previous(self, layer, direction, synset, depth, explored):

		# if hit the max recursion depth
		# return as if it has no more hyper/hypon
		# print(direction, synset, depth)
		if depth == 0:
			oidx = []
			h_prev = torch.zeros(self.hidden_size, 1).to(device)
			c_prev = torch.zeros(self.hidden_size, 1).to(device)

			# print('{}cut-off: {}; depth: {}; direction: {}\n'.format('    ' * (old_depth - depth), synset, depth, direction))
			return oidx, (h_prev, c_prev)

		# check the relationship and pick the synsets from the WN
		if self.relationship == 'hyper_hypon':

			# find the all hyper/hypon synsets of the current nodes by the WN
			if direction == 'up':
				oidx = [hyper.name() for hyper in wn.synset(synset).hypernyms()]
			else:
				oidx = [hypon.name() for hypon in wn.synset(synset).hyponyms()]
		elif self.relationship == 'mer_holo':

			# find the all mer/holo synsets of the current nodes by the WN
			if direction == 'up':
				oidx = [mer.name() for mer in wn.synset(synset).part_meronyms()]
			else:
				oidx = [holo.name() for holo in wn.synset(synset).part_holonyms()]
		else:
			print('Lexical Relationship Not Supported!')
			raise NotImplementedError

		# print(oidx)
		# print('\n')					

		# recursively construct all embedding for the hypers/hypons
		if len(oidx) > 0:
			h_prev, c_prev = [], []
			# print('{}synset: {}; oidx: {} direction: {}\n'.format('    ' * (old_depth - depth), synset, oidx, direction))
			for i in oidx:

				# print('{}{}:'.format('    ' * (old_depth - depth), i))
				# get the h and c for each hyper/hypon
				# (hidden_size)
				h_prev_i, c_prev_i = self._upward_downward(layer, direction, i, depth, {'up': False, 'down': False})
				h_prev.append(h_prev_i)
				c_prev.append(c_prev_i)

			# stack to a new tensor: (hidden_size, num_hyper/num_hypon)
			h_prev = torch.stack(h_prev, 1)
			c_prev = torch.stack(c_prev, 1)

		# if it is a left node (no hyper/hypon), return 0 tensor
		else:
			h_prev = torch.zeros(self.hidden_size, 1).to(device)
			c_prev = torch.zeros(self.hidden_size, 1).to(device)

		# (hidden_size, num_hyper/num_hypon)
		return oidx, (h_prev, c_prev)


class ChildSumGraphLSTM_WordNet(ChildSumGraphLSTM):
	"""A bidirectional extension of child-sum tree LSTMs
		Only runs on the WordNet
	"""

	def _construct_x_t(self, layer, synset):

		# find the x_t
		# when at stacked layer, the input is the previous hidden states (maybe concat when bidirectional)
		# otherwise, x_t comes from the sense embedding
		if layer > 0:

			'''
			# if the previous step did not calculate the hyper
			# this is designed for the stacked version
			# when the recursion order left some of the hypers uncalculated
			if synset not in self.hidden_state[layer - 1]['up']:
				self.hidden_state[layer - 1]['up'][synset] = self._upward_downward((layer - 1), 'up', synset)[0]
			'''

			x_t = torch.cat([self.hidden_state[layer - 1]['up'][synset], self.hidden_state[layer - 1]['down'][synset]])
		else:

			# get the synset (sense) embedding
			synset_idx = self.synset_vocab(synset)
			lookup_tensor = torch.tensor([synset_idx], dtype = torch.long).to(device)
			
			# may add dropout
			if self.dropout:
				dropout = Dropout(p = self.dropout)
				x_t = dropout(self.embedding(lookup_tensor).squeeze(0))
			else:
				x_t = self.embedding(lookup_tensor).squeeze(0)
				# print(x_t.requires_grad)

		# print(x_t.shape)
		return x_t

# test run
def main():

	with open('./data/synset_vocab.pkl', 'rb') as f:
		synset_vocab = pickle.load(f)
	print("Size of synset vocab: {}".format(synset_vocab.idx))

	graph = ChildSumGraphLSTM_WordNet(
		synset_vocab = synset_vocab, 
		relationship = 'hyper_hypon', 
		input_size = 256, 
		hidden_size = 128, 
		num_layers = 2, 
		bidirectional = True, 
		bias = True)

	org = wn.synset('organism.n.01').name()
	worker = wn.synset('worker.n.01').name()
	person = wn.synset('person.n.01').name()
	genus = wn.synset('genus.n.02').name()
	ins = wn.synset('instrumentality.n.03').name()
	synset_list = [org, worker, person, genus, ins]

	start_time = time.time()

	# iterate through all synsets in the WN
	with torch.no_grad():
		for syn in synset_list:
			print(syn)
			hidden_all, (hidden_final, cell_final) = graph(syn, depth = 4)
			print(len(hidden_all.keys()))
			# print(hidden_final.shape, cell_final.shape)

	end_time = time.time()
	print('time: {}'.format((end_time - start_time)))


if __name__ == '__main__':
	main()
