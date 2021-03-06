import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Iterable, defaultdict, Counter
import itertools
import random
import nltk
from nltk.corpus import wordnet as wn
import argparse
import pickle

'''
The specific Synset method is lexname, e.g. wn.synsets('spring')[0].lexname(). 
That should make it really easy to get the suspersenses.
And if you have the synset name–e.g. 'spring.n.01'
you can access the supersense directly: wn.synset('spring.n.01').lexname().
Which returns 'noun.time'.
And wn.synset('spring.n.02').lexname() returns 'noun.artifact'
'''

''' 
build the vocab for the decoder
iterate and tokenize all definitions in the WN for all synsets
'''
class Vocabulary(object):

	"""Simple vocabulary wrapper."""
	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0

	def add_word(self, word):
		if not word in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1

	def __call__(self, word):
		if not word in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]

	def __len__(self):
		return len(self.word2idx)

# build the vocab for the WordNet synsets
def build_vocab_synset():

	i = 0
	total_size = len(set(wn.all_synsets()))

	# Create a vocab wrapper and add some special tokens.
	vocab = Vocabulary()

	# iterate through all definitions in the WN
	for synset in wn.all_synsets():

		# convert '.' to '__' for hashing
		synset = synset.name().replace('.', '__')
		vocab.add_word(synset)

		i += 1
		if i % 1000 == 0:
			print("[{}/{}] synsets done.".format(i, total_size))

	print("Total vocabulary size: {}".format(vocab.idx))
	return vocab

# build the vocab for the WordNet synsets
# those only appears in the SemCor dataset
def build_vocab_synset_SemCor():

	target_file = open("../../WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt", "r")

	# Create a vocab wrapper and add some special tokens.
	vocab = Vocabulary()

	# iterate through all definitions in the SemCor
	for line in target_file:

		# synset and literal definition from the WN
		key = line.replace('\n', '').split(' ')[-1]
		synset = wn.lemma_from_key(key).synset()

		# convert '.' to '__' for hashing
		synset = synset.name().replace('.', '__')
		vocab.add_word(synset)

	# add SemEval synsets
	semeval_file = open("../../WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt", "r")
	for line in semeval_file:
		key = line.replace('\n', '').split(' ')[-1]
		synset = wn.lemma_from_key(key).synset()
		synset = synset.name().replace('.', '__')
		vocab.add_word(synset)

	print("Total vocabulary size: {} {}".format(vocab.idx, len(vocab)))
	return vocab

# build the decoder vocab for SemCor WN definitions
def build_vocab_decoder_SemCor(threshold):
	
	# Create a vocab wrapper and add some special tokens.
	counter = Counter()
	target_file = open("../../WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt", "r")

	# iterate through all definitions in the SemCor
	for line in target_file:

		# synset and literal definition from the WN
		key = line.replace('\n', '').split(' ')[-1]
		synset = wn.lemma_from_key(key).synset()
		definition = synset.definition()
		def_tokens = nltk.tokenize.word_tokenize(definition)
		counter.update(def_tokens)

	# add SemEval synsets
	semeval_file = open("../../WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt", "r")
	for line in semeval_file:
		key = line.replace('\n', '').split(' ')[-1]
		synset = wn.lemma_from_key(key).synset()
		definition = synset.definition()
		def_tokens = nltk.tokenize.word_tokenize(definition)
		counter.update(def_tokens)

	# If the word frequency is less than 'threshold', then the word is discarded.
	words = [word for word, cnt in counter.items() if cnt >= threshold]

	# Create a vocab wrapper and add some special tokens.
	vocab = Vocabulary()
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')

	# Add the words to the vocabulary.
	for i, word in enumerate(words):
		vocab.add_word(word)

	print("Total vocabulary size: {}".format(vocab.idx))
	return vocab

# build the vocab for the WordNet definitions
def build_vocab(threshold):

	"""Build a simple vocabulary wrapper."""
	counter = Counter()
	i = 0
	total_size = len(set(wn.all_synsets()))

	# iterate through all definitions in the WN
	for synset in wn.all_synsets():

		definition = synset.definition()

		'''
		remove special characters
		tokenize the definition string 
		'''
		# tokenize and remove special characters
		# s.translate(str.maketrans('', '', string.punctuation))
		# nltk.tokenize.word_tokenize(wn.synset('dog.n.01').definition())
		def_tokens = nltk.tokenize.word_tokenize(definition.lower())
		counter.update(def_tokens)

		i = i + 1
		if i % 1000 == 0:
			print("[{}/{}] Tokenized the definitions.".format(i, total_size))

	# If the word frequency is less than 'threshold', then the word is discarded.
	words = [word for word, cnt in counter.items() if cnt >= threshold]

	# Create a vocab wrapper and add some special tokens.
	vocab = Vocabulary()
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')

	# Add the words to the vocabulary.
	for i, word in enumerate(words):
		vocab.add_word(word)

	print("Total vocabulary size: {}".format(vocab.idx))
	return vocab

def main(args):

	# vocab = build_vocab(threshold = args.threshold)
	# vocab_synset = build_vocab_synset()
	# vocab_synset_SemCor = build_vocab_synset_SemCor()
	# print(vocab_synset_SemCor('dog__n__01'))
	vocab_decoder_semcor = build_vocab_decoder_SemCor(0)
	vocab_path = args.vocab_path
	with open(vocab_path, 'wb') as f:
		pickle.dump(vocab_decoder_semcor, f)
	print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--vocab_path', type = str, default = '../data/vocab.pkl', 
						help = 'path for saving vocabulary wrapper')
	parser.add_argument('--threshold', type = int, default = 0, 
						help = 'minimum word count threshold')
	args = parser.parse_args()
	from build_vocab import Vocabulary
	main(args)
