import xml.etree.ElementTree as ET
tree = ET.parse('../WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml')
from nltk.corpus import wordnet as wn
import time

corpus = tree.getroot()
# print(corpus)

for text in corpus[1:2]:

	for sentence in text[0:5]:

		tagged_sent = [instance for instance in sentence if instance.tag == 'instance']
		# print([instance.text for instance in tagged_sent])

		if len(tagged_sent) > 0:

			# get all-word definitions, batch_size is the sentence length
			# [batch_size, self.max_length]
			print([instance.text for instance in sentence])
			for instance in tagged_sent:

				# get the sense from the target file with ID
				key = ''
				target_file = open("../WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt", "r")
				for line in target_file:
					if line.startswith(instance.get('id')):
						key = line.replace('\n', '').split(' ')[-1]
				definition = wn.lemma_from_key(key).synset().definition()
				print(instance.text, definition)




