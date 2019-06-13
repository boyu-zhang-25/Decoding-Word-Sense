import xml.etree.ElementTree as ET
tree = ET.parse('../../Downloads/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml')
from nltk.corpus import wordnet as wn 

corpus = tree.getroot()
# print(corpus)

text_num = 0
sentence_num = 0
instance_num = 0

for text in corpus:
	text_num += 1

	for sentence in text:
		# print(sentence.keys())
		sentence_num += 1
		s = [word.text for word in sentence]
		# print(s)

		tagged_sent = [instance for instance in sentence if instance.tag == 'instance']
		# print(tagged_sent)
		# print('\n')

		instance_num += len(tagged_sent)
		for i in tagged_sent:
			if '.' in i.text:
				l = [word.text for word in sentence]
				print(l)
				print(i.text)

print(text_num)
print(sentence_num)
print(instance_num)

f = open("../../Downloads/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt", "r")
s1 = f.readline().replace('\n', '').split(' ')[-1]
print(wn.lemma_from_key(s1).synset().name())
print(f.readline())
