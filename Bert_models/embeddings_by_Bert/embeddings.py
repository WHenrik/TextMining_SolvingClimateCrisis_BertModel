
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, SequentialSampler

from torch.utils.data import TensorDataset
import torch

import numpy as np
import random
import pandas as pd

import time
import datetime


class BertEmbeddings(object):

	
	def __init__(self,key_for_vocab,model_path):
		r"""
    	initializes a model for the extraction of the tweet embeddings.

        Args:
            key_for_vocab (:obj:`str`):
               Keyword which leads to the requested vocabulary
            model_path (:obj:`str`):
               Path that leads to the already pre-trained model
   		"""

		self.key_for_vocab=key_for_vocab
		self.model_path=model_path
		self.model = BertModel.from_pretrained(pretrained_model_name_or_path=self.model_path,output_hidden_states = True)
		self.tokenizer = BertTokenizer.from_pretrained(self.key_for_vocab)

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if device.type=='cuda':
			self.model.cuda();

		self.model.eval();


	def embedd(self,tweets,save_path):
		r"""
    	Extracts the tweet embeddings from the last layers of the Bert model.

        Args:
            tweets (:obj:`Series[str]`):
              Tweets for which the embeddings should be found.
            save_path (:obj:`str`):
               Path where the embeddings should be saved
   		"""
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		input_ids, attention_masks= self.get_encoding(tweets)
		dataset = TensorDataset(input_ids, attention_masks)

		dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = 1)

		dataframe_embeddings=pd.DataFrame()

		embeddings=[]

		seed_val = 42
		random.seed(seed_val)
		np.random.seed(seed_val)
		torch.manual_seed(seed_val)
		torch.cuda.manual_seed_all(seed_val)
		
		# Measure the total training time for the whole run.
		total_t0 = time.time()
		
		print("")
		print('Creating Embeddings...')
		
		# Measure how long the training epoch takes.
		t0 = time.time()
		# For each batch of training data...
		for step, batch in enumerate(dataloader):
		
		    # Progress update every 40 batches.
		    if step % 40 == 0 and not step == 0:
		        # Calculate elapsed time in minutes.
		        elapsed = self.format_time(time.time() - t0)
		            
		        # Report progress.
		        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))
		
		    b_input_ids = batch[0].to(device)
		    b_input_mask = batch[1].to(device)
		
		    embedding=self.get_embeddings(b_input_ids,b_input_mask)
		    embeddings.append(np.array(embedding.cpu().detach()))
		
		    if step%200==0:
		        dataframe_embeddings=pd.concat([dataframe_embeddings,pd.DataFrame(embeddings)])
		        embeddings=[]
		
		dataframe_embeddings=pd.concat([dataframe_embeddings,pd.DataFrame(embeddings)])

		# Measure how long this epoch took.
		embedding_time = self.format_time(time.time() - t0)

		print("")
		print("  Embedding took: {:}".format(embedding_time))

		dataframe_embeddings=dataframe_embeddings.reset_index(drop=True)

		dataframe_embeddings.to_pickle(save_path)

		return dataframe_embeddings



	def get_encoding(self,tweet_set):
    	# Tokenize all of the tweets and map the tokens to thier word IDs.
		input_ids = []
		attention_masks = []		
   		# For every sentence...
		for tweet in tweet_set:
			encoded_dict = self.tokenizer.encode_plus(
   		                    tweet,                      # Tweet to encode.
   		                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
   		                    max_length = 64,           # Pad & truncate all tweet.
   		                    pad_to_max_length = True,
   		                    return_attention_mask = True,   # Construct attn. masks.
   		                    return_tensors = 'pt',     # Return pytorch tensors.
   		                    truncation=True
   		               )
   		
   		    # Add the encoded tweet to the list.    
			input_ids.append(encoded_dict['input_ids'])
   		
   		    # And its attention mask (simply differentiates padding from non-padding).
			attention_masks.append(encoded_dict['attention_mask'])		
   		# Convert the lists into tensors.
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)		
		return input_ids,attention_masks


	def get_embeddings(self,input_ids,attention_masks):
		with torch.no_grad():    #so that no dynamic graph is built
			outputs = self.model(input_ids, attention_masks)
			hidden_states = outputs[2]
			token_vecs = hidden_states[-2][0]
			tweet_embedding = torch.mean(token_vecs, dim=0)
		return tweet_embedding



	
	def format_time(self,elapsed):
	    '''
	    Takes a time in seconds and returns a string hh:mm:ss
	    '''
	    # Round to the nearest second.
	    elapsed_rounded = int(round((elapsed)))
	    
	    # Format as hh:mm:ss
	    return str(datetime.timedelta(seconds=elapsed_rounded))
	

	
