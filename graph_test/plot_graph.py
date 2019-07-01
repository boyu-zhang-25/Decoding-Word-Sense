import torch
import pickle
from new_graph_lstm import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import sample 
import math

def _get_top_list(graph, num):

	if graph == 'hyper_hypon':
		with open('hidden_hyper_hypon.pkl', 'rb') as f:
			all_hiddens = pickle.load(f)
	else:
		with open('hidden_mer_holo.pkl', 'rb') as f:
			all_hiddens = pickle.load(f)

	# get the graph lstm synset vocab that only appears in the SemCor
	with open('./data/synset_vocab_SemCor.pkl', 'rb') as f:
		synset_vocab_SemCor = pickle.load(f)
	print("Size of synset vocab in SemCor: {}".format(synset_vocab_SemCor.idx))

	# pick the top graphs (with most num of nodes) as samples
	top_list = []
	starting_node = []
	for i in range(0, num):  
		max1 = dict()
		for j in range(len(all_hiddens)):      
			if len(all_hiddens[j].keys()) > len(max1.keys()): 
				max1 = all_hiddens[j]

				# keep track of the starting node
				m = j + i
				starting_node.append(synset_vocab_SemCor.idx2word.get(j).replace('__', '.'))

		all_hiddens.remove(max1)
		top_list.append(max1)

	file = 'results_' + graph + '_top' + str(num) + '.pkl'
	# print(len(top_list))
	results = (top_list, starting_node)
	with open(file, 'wb') as f:
		pickle.dump(results, f)

def _get_tSNE(file):

	with open(file, 'rb') as f:
		(top_list, starting_node) = pickle.load(f)

	# one plot for each subgraph
	graph_type = file.split('.')[0]
	tSNE_list = []
	# print(len(top_list))
	for embedding_dict in top_list:

		# convert each subgraph to numpy with index reference
		keys = []
		tensors = []
		print('num of emb in current subgraph: {}'.format(len(embedding_dict.keys())))
		for key, val in embedding_dict.items():
			keys.append(key)
			# print(val.cpu().numpy().size)
			tensors.append(val.cpu().numpy())
		tensors = np.asarray(tensors)
		print('before tSNE: {}'.format(tensors.shape))

		# fit tSNE
		tensors_embedded = TSNE(n_components = 2).fit_transform(tensors)
		print('after tSNE: {}'.format(tensors_embedded.shape))

		# put back to the dict
		for j, key in enumerate(keys):
			embedding_dict.update({key: tensors_embedded[j, :]})

		tSNE_list.append(embedding_dict)

	# store new embeddings to pkl
	file = 'tSNE_' + graph_type + '.pkl'
	results = (tSNE_list, starting_node)
	with open(file, 'wb') as f:
		pickle.dump(results, f)

def _plot_senses(tSNE_file, sample_percent, alpha, labeled):

	with open(tSNE_file, 'rb') as f:
		(tSNE_list, starting_node) = pickle.load(f)
	# tSNE_list = tSNE_list[]

	# one color for each subgraph
	colors = cm.rainbow(np.linspace(0, 1, len(tSNE_list)))

	# convert the embeddings in each subgraph back in order
	for embedding_dict, color, k in zip(tSNE_list, colors, range(len(tSNE_list))):

		plt.figure(k, figsize=(16, 9))

		# mark the starting point
		start_node = starting_node[k].replace('.', '__')
		start_index = list(embedding_dict.keys()).index(start_node)
		start_node_x = embedding_dict.get(start_node)[0]
		start_node_y = embedding_dict.get(start_node)[1]

		# select a subset of them to plot
		num_sample = math.floor(len(embedding_dict.keys()) * sample_percent)
		keys = sample(embedding_dict.keys(), num_sample)
		tensors = [embedding_dict.get(key) for key in keys]

		# scatter plot for each subgraph after stacking
		embeddings = np.stack(tensors)
		x = embeddings[:, 0]
		y = embeddings[:, 1]
		plt.scatter(x, y, c = color, alpha = alpha)
		plt.scatter(start_node_x, start_node_y, c = 'black', alpha = 1.0, marker='^', markersize = 12)

		for i, sense in enumerate(keys):
			sense = sense.replace('__', '.')
			if labeled:
				plt.annotate(sense, alpha = 0.5, xy = (x[i], y[i]), xytext = (5, 2), textcoords = 'offset points', ha = 'right', va = 'bottom', size = 8)

		# plot settings for each subgraph
		if labeled:
			title = tSNE_file.split('_')[-2] + '_' + tSNE_file.split('_')[-1].split('.')[0] + '_' + start_node + '_' + 'labeled'
		else:
			title = tSNE_file.split('_')[-2] + '_' + tSNE_file.split('_')[-1].split('.')[0] + '_' + start_node
		plt.legend(loc = 4)
		plt.title(title)
		plt.grid(True)
		plt.savefig(title, format='png', dpi=150, bbox_inches='tight')
		# plt.show()

# _get_top_list(graph = 'hyper_hypon', num = 5)
# _get_tSNE('results_hyper_hypon_top5.pkl')
_plot_senses(tSNE_file = 'tSNE_results_hyper_hypon_top5.pkl', sample_percent = 0.2, alpha = 0.7, labeled = True)

	