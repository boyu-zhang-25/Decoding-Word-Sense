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

class ChildSumGraphLSTM(RNNBase):
	"""A bidirectional extension of child-sum tree LSTMs that work on graphs
	For each node, the embedding is calculated recursively 
	from all connected branches in the graph.
	It supports bidirection and stacked layers.

	This class cannot be instantiated directly. 
	Instead, the following subclasses runs on the WordNet:
	  - ChildSumGraphLSTM_WordNet
	"""

	__metaclass__ = ABCMeta

	def __init__(self, synset_vocab, *args, **kwargs):
		super(ChildSumGraphLSTM, self).__init__('LSTM', *args, **kwargs)

		# lru_cache is normally used as a decorator, but that usage
		# leads to a global cache, where we need an instance specific
		# cache
		self._get_parameters = lru_cache()(self._get_parameters)

		# all synset embeddings and their indices
		self.synset_vocab = synset_vocab
		self.embedding = Embedding(synset_vocab.idx, self.input_size)

	@staticmethod
	def nonlinearity(x):
		return torch.tanh(x)

	'''
	given a synset (in '__' due to hashing)
	use all its hypers and hypons to generate the new node embedding
	recursively over the whole graph
	'''
	def forward(self, inputs, synset):
		"""
		Parameters
		----------
		inputs : torch.Tensor
			a 2D (steps x embedding dimension) or a 3D tensor (steps x
			batch dimension x embedding dimension); the batch
			dimension must always have size == 1, since this module
			does not support minibatching

		synset: the name of a 'nltk.corpus.reader.wordnet.Synset'
			the current synset (node)
			must be coverted to string by wn.synset().name()
			and convert '.' to '__' due to hashing

		Due to the nltk.corpus.wordnet package support,
		user does not have to provide the graph to the forward function

		Returns
		-------
		hidden_all: torch.Tensor
			the updated hidden states of all connected nodes along the resursion process
		hidden_final, cell_final: torch.Tensor
			the final hidden state and cell state of the target synset node.
		"""

		# self._validate_inputs(inputs)

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
			self._upward_downward(layer, 'up', inputs, synset)
			# self._upward_downward(layer, 'down', inputs, synset)

		# convert '__' to '.' due to hashing
		synset = synset.replace('.', '__')

		# the hidden states of all connected hyper and hypons during the recursion
		# the final hidden state and cell state of the current synset
		hidden_up = self.hidden_state[self.num_layers - 1]['up']
		if self.bidirectional:
			hidden_down = self.hidden_state[self.num_layers - 1]['down']
			hidden_all = [torch.cat([hidden_up[key], hidden_down[key]])
						  for key in self.hidden_state[self.num_layers - 1]['up'].keys()]

			hidden_final = torch.cat([hidden_up[synset], hidden_down[synset]])
			cell_final = torch.cat([self.cell_state[self.num_layers - 1]['up'][synset], self.cell_state[self.num_layers - 1]['down'][synset]])

		else:
			hidden_all = [hidden_up[key] for key in self.hidden_state[self.num_layers - 1]['up'].keys()]
			hidden_final = hidden_up[synset]
			cell_final = self.cell_state[self.num_layers - 1]['up'][synset]

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
	def _upward_downward(self, layer, direction, inputs, synset):

		# print(direction)
		# convert '__' to '.' due to hashing
		synset = synset.replace('.', '__')

		# check to see whether this node has been computed on this
		# layer in this direction, if so short circuit the rest of
		# this function and return that result
		# very useful in cyclic graphs
		if synset in self.hidden_state[layer][direction]:
			# print('short-circuit\n')
			h_t = self.hidden_state[layer][direction][synset]
			c_t = self.cell_state[layer][direction][synset]

			return h_t, c_t

		# get the current node x_t from the embedding
		x_t = self._construct_x_t(layer, inputs, synset)

		# construct the hyper and hypon embedding recursively
		# h_prev, c_prev: (hidden_size, num_hyper/num_hypon)
		# convert '__' to '.' due to hashing
		synset = synset.replace('__', '.')
		oidx, (h_prev, c_prev) = self._construct_previous(layer, direction,
														  inputs, synset)

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
		# improve efficiency when short-circuit
		# convert '.' back to '__' due to hashing
		synset = synset.replace('.', '__')
		self.hidden_state[layer][direction][synset] = h_t
		self.cell_state[layer][direction][synset] = c_t

		# get the hypon
		if self.bidirectional and direction == 'up':
			self._upward_downward(layer, 'down', inputs, synset)

		# (hidden_size)
		return h_t, c_t

	# validate the input shape
	# only support SGD with batch size of 1
	def _validate_inputs(self, inputs):
		if len(inputs.size()) == 3:
			self._has_batch_dimension = True
			try:
				assert inputs.size()[1] == 1
			except AssertionError:
				msg = 'ChildSumTreeLSTM assumes that dimension 1 of'
				msg += 'inputs is a batch dimension and, because it'
				msg += 'does not support minibatching, this dimension'
				msg += 'must always have size == 1'
				raise ValueError(msg)
		elif len(inputs.size()) == 2:
			self._has_batch_dimension = False
		else:
			msg = 'inputs must be 2D or 3D (with dimension 1 being'
			msg += 'a batch dimension)'
			raise ValueError(msg)

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
	def _construct_x_t(self, layer, inputs, synset):
		raise NotImplementedError

	'''
	given the current synset (node, in '.') and the direction
	recursively find all its hyper or hypon embeddings
	'''
	def _construct_previous(self, layer, direction, inputs, synset):

		# find the all hyper/hypon synsets of the current nodes by the WN
		if direction == 'up':
			oidx = [hyper.name() for hyper in wn.synset(synset).hypernyms()]
		else:
			oidx = [hypon.name() for hypon in wn.synset(synset).hyponyms()]

		# recursively construct all embedding for the hypers/hypons
		if len(oidx) > 0:
			h_prev, c_prev = [], []

			for i in oidx:

				# get the h and c for each hyper/hypon
				# (hidden_size)
				h_prev_i, c_prev_i = self._upward_downward(layer, direction, inputs, i)
				h_prev.append(h_prev_i)
				c_prev.append(c_prev_i)

			# stack to a new tensor: (hidden_size, num_hyper/num_hypon)
			h_prev = torch.stack(h_prev, 1)
			c_prev = torch.stack(c_prev, 1)

		# if it is a left node (no hyper/hypon), return 0 tensor
		elif torch.cuda.is_available():
			h_prev = torch.zeros(self.hidden_size, 1).to(device)
			c_prev = torch.zeros(self.hidden_size, 1).to(device)
		else:
			h_prev = torch.zeros(self.hidden_size, 1)
			c_prev = torch.zeros(self.hidden_size, 1)

		# (hidden_size, num_hyper/num_hypon)
		return oidx, (h_prev, c_prev)


class ChildSumGraphLSTM_WordNet(ChildSumGraphLSTM):
	"""A bidirectional extension of child-sum tree LSTMs
		Only runs on the WordNet
	"""

	def _construct_x_t(self, layer, inputs, synset):

		# find the x_t
		# when at stacked layer, the input is the previous hidden state
		# otherwise, x_t comes from the sense embedding
		if layer > 0 and self.bidirectional:

			# if the previous step did not calculate the hyper
			# this is designed for the stacked version
			# when the recursion order left some of the hypers uncalculated
			if synset not in self.hidden_state[layer - 1]['up']:
				self.hidden_state[layer - 1]['up'][synset] = self._upward_downward((layer - 1), 'up', inputs, synset)[0]

			x_t = torch.cat([self.hidden_state[layer - 1]['up'][synset], self.hidden_state[layer - 1]['down'][synset]])

		elif layer > 0:
			x_t = self.hidden_state[layer - 1]['up'][synset]
		else:

			# get the synset (sense) embedding
			synset_idx = self.synset_vocab(synset)
			lookup_tensor = torch.tensor([synset_idx], dtype = torch.long)
			
			# may add dropout
			if self.dropout:
				dropout = Dropout(p = self.dropout)
				x_t = dropout(self.embedding(lookup_tensor))
			else:
				x_t = self.embedding(lookup_tensor)

		return x_t

# test run
def main():

	with open('./data/synset_vocab.pkl', 'rb') as f:
		synset_vocab = pickle.load(f)
    print("Size of synset vocab: {}".format(synset_vocab.idx))

    graph = ChildSumGraphLSTM_WordNet(synset_vocab = synset_vocab, input_size = 1, hidden_size = 1, num_layers = 2, bidirectional = True, bias = True)
	synset = wn.synset('dog.n.01').name()

	import time
	start_time = time.time()
	output = graph('input', synset)
	end_time = time.time()
	print('time: {}'.format((end_time - start_time)))

if __name__ == '__main__':
	main()