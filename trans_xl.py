import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel
import pickle

# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary from wikitext 103)
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

# Tokenized input
text_1 = "what do we need to make"
text_2 = "in order to make it or to"
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)
print(tokenized_text_1, tokenized_text_2)

# get the decoder vocab
with open('./data/vocab.pkl', 'rb') as f:
	vocab = pickle.load(f)
	print("Size of vocab: {}".format(vocab.idx))

words_in_order = [vocab.idx2word.get(idx) for idx in range(vocab.idx)]
# print(words_in_order)

# see how many unk words are there
num = []
for key, value in vocab.idx2word.items():
	if tokenizer.convert_tokens_to_ids([value]) == [24]:
		num.append(value)
print(num)
# print(len(num))

# print(tokenizer.convert_ids_to_tokens([24]))
# Convert token to vocabulary indices
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

# Load pre-trained model (weights)
model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor_1 = tokens_tensor_1.to(device)
tokens_tensor_2 = tokens_tensor_2.to(device)
model.to(device)

with torch.no_grad():
    # Predict all tokens
    predictions_1, mems_1 = model(tokens_tensor_1)
    # We can re-use the memory cells in a subsequent call to attend a longer context
    predictions_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

# get the predicted last token
print(predictions_2[0, -1, :].shape)
predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
assert predicted_token == 'who'

