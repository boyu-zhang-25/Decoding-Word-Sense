# randomly initialize weights
def init_weights(m):
	for name, param in m.named_parameters():
		nn.init.uniform_(param.data, -0.08, 0.08)
		
# model.apply(init_weights)

if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# self._model = nn.DataParallel(self._model)

# supports batch input
def train(self, train_X, train_Y, train_idx, dev_X, dev_Y, dev_idx, **kwargs):

	# train_Y is the annotator response
	self.train_X, self.train_Y = train_X, train_Y
	self.dev_X, self.dev_Y = dev_X, dev_Y
		
	self._initialize_trainer_model()  
	# old = self._model.definition_embeddings['spring'].clone().detach()

	# trainer setup
	parameters = [p for p in self._model.parameters() if p.requires_grad]
	optimizer = self.optimizer(parameters, weight_decay = self.optim_wt_decay, **kwargs)

	num_train = len(self.train_X)
	# num_dev = len(self.dev_X)
	
	# dev_accs = []
	best_loss = float('inf')
	best_r = -float('inf')
	train_losses = []
	dev_losses = []
	dev_rs = []
	bad_count = 0
	
	for epoch in range(self.epochs):
		
		# loss for the current iteration
		batch_losses = []

		# Turn on training mode which enables dropout.
		# self._model.train()			
		tqdm.write("[Epoch: {}/{}]".format((epoch + 1), self.epochs))
		
		# time print
		pbar = tqdm_n(total = num_train)
		s_list = []

		# SGD batch = 1
		for idx, sentence in enumerate(self.train_X):
			
			# Zero grad
			optimizer.zero_grad()

			# the target word
			word_idx = train_idx[idx]
			word_lemma = '____' + sentence[word_idx]

			# model output
			sense_vec = self._model.forward(sentence, word_idx)
			# print(sense_vec.type())
			# s_list.append(sense_vec)

			# calculate loss pair-wise: sense vector and definition vector
			# accumulative loss
			loss = torch.zeros(1).to(self.device)

			# check all definitions in the annotator response for the target word
			for i, response in enumerate(self.train_Y[idx]):

				# slice the particular definition for gradient calculation
				definition_vec = self._model.definition_embeddings[word_lemma][:, i].view(1, -1)
				# print(sense_vec.type())

				# find the supersense
				synset = self.all_senses[word_lemma][i]
				supersense = wn.synset(synset).lexname().replace('.', '_')
				supersense_vec = self._model.supersense_embeddings[supersense].view(1, -1)

				if response:

					# if annotator response is True: increase the cosine similarity
					# loss between sense embeddings and the definition embeddings
					y_ = torch.ones(1).to(self.device)
					loss += self.loss(sense_vec, definition_vec, y_)

					# loss between the supersense and the sensen embeddings
					loss += self.loss(sense_vec, supersense_vec, y_)

					# loss between the supersense and the definition embeddings
					# they should always be similar
					loss += self.loss(definition_vec, supersense_vec, y_)

				else:

					# if annotator response is False
					# decrease the cosine similarity
					y_ = torch.ones(1).to(self.device)
					y_neg = -torch.ones(1).to(self.device)
					loss += self.loss(sense_vec, definition_vec, y_neg)
					loss += self.loss(sense_vec, supersense_vec, y_neg)

					# loss between the supersense and the definition embeddings
					# they should always be similar
					loss += self.loss(definition_vec, supersense_vec, y_)

			# individual definition tensor gradient update
			# also backprop the accumulative loss for the predicted sense embeddings
			loss.backward()
			optimizer.step()

			# record training loss for each example
			current_loss = loss.detach().item()
			batch_losses.append(current_loss)
			pbar.update(1)
				
		pbar.close()
		
		# calculate the training loss of the current epoch
		curr_train_loss = np.mean(batch_losses)
		print("Epoch: {}, Mean Training Loss: {}".format(epoch + 1, curr_train_loss))
		train_losses.append(curr_train_loss)

		# dev loss of the current epoch
		curr_dev_loss = np.mean(self.dev_loss(dev_X, dev_Y, dev_idx))
		print("Epoch: {}, Mean Dev Loss: {}".format(epoch + 1, curr_dev_loss))
		dev_losses.append(curr_dev_loss)

		# save the best model by dev 
		if curr_dev_loss <= best_loss:
			with open(self.best_model_file, 'wb') as f:
				torch.save(self._model.state_dict(), f)
			best_loss = curr_dev_loss
		
		# early stopping
		'''
		if epoch:
			if (abs(curr_train_loss - train_losses[-1]) < 0.0001):
				break
		'''
	return train_losses, dev_losses, dev_rs

# for dev 
def dev_loss(self, dev_X, dev_Y, dev_idx):

	dev_losses = []

	for idx, sentence in enumerate(dev_X):

		# the target word
		word_idx = dev_idx[idx]
		word_lemma = '____' + sentence[word_idx]

		# model output
		sense_vec = self._model.forward(sentence, word_idx)
		loss = 0.0

		# check all definitions in the annotator response for the target word
		# dev set only contains known words
		for i, response in enumerate(dev_Y[idx]):

			# slice the particular definition for gradient calculation
			definition_vec = self._model.definition_embeddings[word_lemma][:, i].view(1, -1)
					
			# find the supersense
			synset = self.all_senses[word_lemma][i]
			supersense = wn.synset(synset).lexname().replace('.', '_')
			supersense_vec = self._model.supersense_embeddings[supersense].view(1, -1)

			y_ = torch.ones(1).to(self.device)
			y_neg = -torch.ones(1).to(self.device)

			if response:

				loss += self.loss(sense_vec, definition_vec, y_)
				loss += self.loss(sense_vec, supersense_vec, y_)
				loss += self.loss(definition_vec, supersense_vec, y_)

			else:
				loss += self.loss(sense_vec, definition_vec, y_neg)
				loss += self.loss(sense_vec, supersense_vec, y_neg)
				loss += self.loss(definition_vec, supersense_vec, y_)

		# record training loss for each example
		dev_loss = loss.detach().item()
		dev_losses.append(dev_loss)

	return dev_losses