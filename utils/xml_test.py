import xml.etree.ElementTree as ET
tree = ET.parse('../../WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml')
from nltk.corpus import wordnet as wn
import time
import math

corpus = tree.getroot()

sentence_num = 0
i = 0
for text in corpus[30:40]:

	i += 1
	sentence_num += len(text)

	if i % 10 == 0:
		print("[{}/{}] subcorpus done.".format(i, len(corpus)))

	'''
	for sentence in text:

		tagged_sent = [instance for instance in sentence if instance.tag == 'instance']
		# print([instance.text for instance in tagged_sent])

		if len(tagged_sent) > 0:

			# get all-word definitions, batch_size is the sentence length
			# [batch_size, self.max_length]
			print([instance.text for instance in sentence])
	'''
print(sentence_num)





