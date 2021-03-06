import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel
import pickle
from io import open

# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary from wikitext 103)
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

# Tokenized input
text_1 = "what do we need"
text_2 = "in order to finish"
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)
# print(tokenized_text_1, tokenized_text_2)

# get the decoder vocab
with open('./data/vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)
	# print(vocab('<unk>'))
	print("Size of vocab: {}".format(vocab.idx))

word_idx_in_order = [tokenizer.convert_tokens_to_ids([vocab.idx2word.get(idx)])[0] for idx in range(vocab.idx)]
print(len(word_idx_in_order))

# see how many unk words are there
'''
num = []
for key, value in vocab.idx2word.items():
	if tokenizer.convert_tokens_to_ids([value]) == [24]:
		num.append(value)
print(num)
print(len(num))
'''

# print(tokenizer.convert_ids_to_tokens([24]))
# Convert token to vocabulary indices
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
# print(indexed_tokens_1)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor(indexed_tokens_1)
tokens_tensor_2 = torch.tensor(indexed_tokens_2)

final_tensor = torch.stack((tokens_tensor_1, tokens_tensor_2), 0)
print(final_tensor.shape)

# Load pre-trained model (weights)
model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
model.eval()
'''
with open('trans_vocab.txt', 'w') as f:
	for i in range(267735):
		token = tokenizer.convert_ids_to_tokens([i])[0]
		f.write("%s\n" % token)
'''
# print(tokenizer.convert_tokens_to_ids(['']))
# If you have a GPU, put everything on cuda
tokens_tensor_1 = tokens_tensor_1.to(device)
tokens_tensor_2 = tokens_tensor_2.to(device)
final_tensor = final_tensor.to(device)
model.to(device)

with torch.no_grad():
    # Predict all tokens
    # predictions_1, mems_1 = model(tokens_tensor_1)
    # We can re-use the memory cells in a subsequent call to attend a longer context
    # target = tokens_tensor_1
    # predictions_2, mems_2 = model(tokens_tensor_2, mems = mems_1)
    predictions_2, mems_2 = model(final_tensor)
    print(predictions_2.shape)

# get the predicted last token
# print(predictions_2[0, -1, :].shape)
predicted_index = torch.argmax(predictions_2[:, -1, :], dim = 1, keepdim = True)
print(predicted_index.shape)
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index[0]])
print(predicted_token)
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index[1]])
print(predicted_token)
# get subset vocab for our model
our_prediction = torch.zeros(len(word_idx_in_order))
for our_idx, xl_idx in enumerate(word_idx_in_order):
	our_prediction[our_idx] = predictions_2[0, -1, xl_idx]
our_index = torch.argmax(our_prediction).item()
our_token = vocab.idx2word.get(our_index)
print(our_token)
print(vocab(predicted_token))
