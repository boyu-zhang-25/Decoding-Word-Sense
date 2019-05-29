import sys
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
		counter.update(tokens)

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
	return vocab

def main(args):
	vocab = build_vocab(threshold = args.threshold)
	vocab_path = args.vocab_path
	with open(vocab_path, 'wb') as f:
		pickle.dump(vocab, f)
	print("Total vocabulary size: {}".format(len(vocab)))
	print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--vocab_path', type = str, default = './data/vocab.pkl', 
						help = 'path for saving vocabulary wrapper')
	parser.add_argument('--threshold', type = int, default = 4, 
						help = 'minimum word count threshold')
	args = parser.parse_args()
	main(args)
