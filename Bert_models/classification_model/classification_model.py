
from transformers import BertForSequenceClassification, AdamW, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch

import numpy as np
import random
import pandas as pd

import time
import datetime

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class BertClassificationModel(object):
	r"""
    Initializes the Bert classification model that was used to predict in which domain (industry, news, politician, none) a tweet can be assigned

        Args:
            key_for_vocab (:obj:`str`):
               Keyword which leads to the requested vocabulary
            model_path (:obj:`str`):
               Path that leads to the already pre-trained model
            num_labels (:obj:`int`):
               Number of classes to predict
            show_parameter (:obj:`bool`):
               if True: the structure of the model will be printed
    """
	
	def __init__(self,key_for_vocab,model_path,num_labels,show_parameter=False):
		self.key_for_vocab=key_for_vocab
		self.num_labels=num_labels
		self.model_path=model_path
		self.model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=self.model_path, num_labels = self.num_labels, output_attentions = False, output_hidden_states = False)
		self.tokenizer = BertTokenizer.from_pretrained(self.key_for_vocab)

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if device.type=='cuda':
			self.model.cuda();

		if show_parameter:
			# Get all of the model's parameters as a list of tuples.
			params = list(self.model.named_parameters())

			print('The BERT model has {:} different named parameters.\n'.format(len(params)))
			print('==== Embedding Layer ====\n')
			for p in params[0:5]:
				print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
			print('\n==== First Transformer ====\n')
			for p in params[5:21]:
				print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
			print('\n==== Output Layer ====\n')
			for p in params[-4:]:
				print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


	def fit(self,data_train,save_path,data_test=None, batch_size=64, epochs=4):
		r"""
   		Trains the model on the given learning data by using the trainer class

        Args:
            data_train (:obj:`DataFrame[Series[str],Series[str]]`):
               Training data for the model
            save_path (:obj:`str`):
               Path where the model should be saved
            data_test (:obj:`DataFrame[Series[str],Series[str]]`):
               Test data for the model
            batch_size (:obj:`int`):
               Size of batches
            epochs (:obj:`int`):
               Number of epochs the model will be trained (and evaluated)
    	"""

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		input_ids_train, attention_masks_train= self.tokenize(data_train.text)

		# Convert the lists into tensors.
		input_ids_train = torch.cat(input_ids_train, dim=0)
		attention_masks_train = torch.cat(attention_masks_train, dim=0)
		labels_train = torch.tensor(data_train.label.to_numpy())

		if data_test is not None:
			input_ids_test, attention_masks_test= self.tokenize(data_test.text)

			input_ids_test = torch.cat(input_ids_test, dim=0)
			attention_masks_test = torch.cat(attention_masks_test, dim=0)
			labels_test = torch.tensor(data_test.label.to_numpy())

			val_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
			validation_dataloader = DataLoader( val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)

		# Combine the training inputs into a TensorDataset.
		train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)	
		train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)

		optimizer = AdamW(self.model.parameters(),
				lr = 2e-5, # should be between 2e-5 and 5e-5
                eps = 1e-8 
                )
		total_steps = len(train_dataloader) * epochs

		# Create the learning rate scheduler.
		scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

		seed_val = 42

		random.seed(seed_val)
		np.random.seed(seed_val)
		torch.manual_seed(seed_val)
		torch.cuda.manual_seed_all(seed_val)
		
		training_stats = []
		
		# Measure the total training time for the whole run.
		total_t0 = time.time()
		
		# For each epoch...
		for epoch_i in range(0, epochs):
		    
		    # ========================================
		    #               Training
		    # ========================================
		    
		    # Perform one full pass over the training set.
		
		    print("")
		    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
		    print('Training...')
		
		    # Measure how long the training epoch takes.
		    t0 = time.time()
		
		    # Reset the total loss for this epoch.
		    total_train_loss = 0
		
		    # Put the model into training mode.
		    self.model.train()
		
		    # For each batch of training data...
		    for step, batch in enumerate(train_dataloader):
		
		        # Progress update every 40 batches.
		        if step % 40 == 0 and not step == 0:
		            # Calculate elapsed time in minutes.
		            elapsed = self.format_time(time.time() - t0)
		            
		            # Report progress.
		            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

		        b_input_ids = batch[0].to(device)
		        b_input_mask = batch[1].to(device)
		        b_labels = batch[2].to(device)
		
		        # clear any previously calculated gradients before performing backward pass. 
		        self.model.zero_grad()        
		
		        # Perform a forward pass 
		        loss, logits = self.model(b_input_ids, 
		                             token_type_ids=None, 
		                             attention_mask=b_input_mask, 
		                             labels=b_labels)
		
		        total_train_loss += loss.item()
		
		        # Perform a backward pass to calculate the gradients.
		        loss.backward()
		
		        # Clip the norm of the gradients to 1.0 to help prevent the "exploding gradients" problem.
		        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
		
		        # Update parameters and take a step using the computed gradient.
		        optimizer.step()
		
		        # Update the learning rate.
		        scheduler.step()
		
		    # Calculate the average loss over all of the batches.
		    avg_train_loss = total_train_loss / len(train_dataloader)            
		    
		    # Measure how long this epoch took.
		    training_time = self.format_time(time.time() - t0)
		
		    print("")
		    print("  Average training loss: {0:.2f}".format(avg_train_loss))
		    print("  Training epoch took: {:}".format(training_time))
		    

		    if data_test is not None:    
		    	# ========================================
		    	#               Validation
		    	# ========================================
		    	# After the completion of each training epoch, measure our performance on
		    	# our validation set.
			
		    	print("")
		    	print("Running Validation...")
			
		    	t0 = time.time()
			
		    	## Put the model in evaluation mode --> dropout layers behave differently during evaluation.
		    	self.model.eval()
			
		    	# Tracking variables 
		    	total_eval_accuracy = 0
		    	total_eval_loss = 0
		    	nb_eval_steps = 0
			
		    	## Evaluate data for one epoch
		    	for batch in validation_dataloader:
		    	    
		    	    b_input_ids = batch[0].to(device)
		    	    b_input_mask = batch[1].to(device)
		    	    b_labels = batch[2].to(device)
		    	    
		    	    # No compute graph will be constructed, since this is only needed for backprop (training).
		    	    with torch.no_grad():        
			
		    	        # Forward pass, calculate logit predictions.
		    	        (loss, logits) = self.model(b_input_ids, 
		    	                               token_type_ids=None, 
		    	                               attention_mask=b_input_mask,
		    	                               labels=b_labels)
		    	        
		    	    # Accumulate the validation loss.
		    	    total_eval_loss += loss.item()
			
		    	   # Move logits and labels to CPU
		    	    logits = logits.detach().cpu().numpy()
		    	    label_ids = b_labels.to('cpu').numpy()
			
		    	    # Calculate the accuracy for this batch of test tweet, and accumulate it over all batches.
		    	    total_eval_accuracy += self.flat_accuracy(logits, label_ids)
		    	    
			
		    	## Report the final accuracy for this validation run.
		    	avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
		    	print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
			
		    	# Calculate the average loss over all of the batches.
		    	avg_val_loss = total_eval_loss / len(validation_dataloader)
		    	
		    	# Measure how long the validation run took.
		    	validation_time = self.format_time(time.time() - t0)
		    	
		    	print("  Validation Loss: {0:.2f}".format(avg_val_loss))
		    	print("  Validation took: {:}".format(validation_time))
			
		    	# Record all statistics from this epoch.
		    	training_stats.append(
		    	    {
		    	        'epoch': epoch_i + 1,
		    	        'Training Loss': avg_train_loss,
		    	        'Valid. Loss': avg_val_loss,
		    	        'Valid. Accur.': avg_val_accuracy,
		    	        'Training Time': training_time,
		    	        'Validation Time': validation_time
		    	    }
		    	)
		
		print("")
		print("Training complete!")
		
		print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))

		#save model
		self.model.save_pretrained(save_path)

	def predict(self,tweet,show_result=True):
		r"""
   		Lets the model assign a class to a tweet

        Args:
            data_train (:obj:`str`):
               Tweet for which the class should be predicted
            show_result (:obj:`bool`):
               Prints the predicted class

   		Returns:
        	:obj:`str`: predicted class
    	"""


		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		input_id_test=[]
		attentition_mask_test=[]
	
	
    	# For test tweet...
		encoded_dict = self.tokenizer.encode_plus(
    	                    tweet,                      # tweet to encode.
    	                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
    	                    max_length = 64,           # Pad & truncate all tweets.
    	                    pad_to_max_length = True,
    	                    return_attention_mask = True,   # Construct attn. masks.
    	                    return_tensors = 'pt',     # Return pytorch tensors.
    	                    truncation=True
    	               )
    	
    	# Add the encoded tweets to the list.    
		input_id_test.append(encoded_dict['input_ids'])
    	# And its attention mask (simply differentiates padding from non-padding).
		attentition_mask_test.append(encoded_dict['attention_mask'])
	
		input_id = torch.cat(input_id_test, dim=0).to(device)
		attention_mask = torch.cat(attentition_mask_test, dim=0).to(device)
	
		label_test = self.model(input_id, 
    	                               token_type_ids=None, 
    	                               attention_mask=attention_mask,
    	                               )
	
		end_prediction_choices=['None', 'industry', 'politician', 'news']
		prediction= end_prediction_choices[np.argmax(np.array(label_test[0].cpu().detach()))]

		if show_result:
			print(tweet + "\n")
			print("Dieser Tweet kann " + prediction + " zugeordnet werden")

		return(prediction)


	def confusion_matrix(self,data_test):
		r"""
   		Visualizes the confusion matrix

        Args:
            data_test (:obj:`str`):
               Tweets for which the confusion matrix should be created.
    	"""



		real_labels = data_test.label
		real_labels

		dtype = pd.CategoricalDtype(['None', 'industry', 'politician', 'news'], ordered=True)
		real_labels_strings = pd.Categorical.from_codes(codes=real_labels, dtype=dtype)
		
		data_test=data_test.reset_index(drop=True)
		predicted_labels = [self.predict(test_tweet,show_result=False) for test_tweet in data_test['text']]    #brauch nur 
		
		def show_confusion_matrix(confusion_matrix):
			sns.set(font_scale=2)
			plt.figure(figsize=[18,12])
			hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
			hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
			hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
			plt.ylabel('True label')
			plt.xlabel('Predicted label')
			plt.show()
		
		cm = confusion_matrix(real_labels_strings, predicted_labels)
		df_cm = pd.DataFrame(cm, index=['Other', 'Industry', 'News', 'Politician'], columns=['Other', 'Industry', 'News', 'Politician'])
		show_confusion_matrix(df_cm)



	def tokenize(self,tweets_to_tokenize):
		# Tokenize all of the tweets and map the tokens to thier word IDs.
		input_ids = []         
		attention_masks = []       
	
		# For every tweet...
		for tweet in tweets_to_tokenize:
			encoded_dict = self.tokenizer.encode_plus(
	                        tweet,                      # tweets to encode.
	                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
	                        max_length = 64,           # Pad & truncate all tweets.
	                        pad_to_max_length = True,
	                        return_attention_mask = True,   # Construct attn. masks.
	                        return_tensors = 'pt',     # Return pytorch tensors.
	                        truncation=True
	                   )
	    
	    	# Add the encoded tweet to the list.    
			input_ids.append(encoded_dict['input_ids'])
	    
			# And its attention mask (simply differentiates padding from non-padding).
			attention_masks.append(encoded_dict['attention_mask'])
	
		return input_ids, attention_masks
	
	def format_time(self,elapsed):
	    '''
	    Takes a time in seconds and returns a string hh:mm:ss
	    '''
	    # Round to the nearest second.
	    elapsed_rounded = int(round((elapsed)))
	    
	    # Format as hh:mm:ss
	    return str(datetime.timedelta(seconds=elapsed_rounded))
	
	# Function to calculate the accuracy of our predictions vs labels
	def flat_accuracy(self,preds, labels):
	    pred_flat = np.argmax(preds, axis=1).flatten()
	    labels_flat = labels.flatten()
	    return np.sum(pred_flat == labels_flat) / len(labels_flat)
	
