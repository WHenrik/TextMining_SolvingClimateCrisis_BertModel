from transformers import BertTokenizer, BertForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from utils.textfile_creator import create_txt_file

import os
import torch


class BertLanguageModel(object):
    """
    Initializes the Bert language model that was used to learn the climate-specific context from the tweets.
    
        Args:
            key_for_vocab (:obj:`str`):
                Keyword which leads to the requested vocabulary
            model_path (:obj:`str`):
                Path that leads to the already pre-trained model
    """
    def __init__(self,key_for_vocab,model_path):
        self.key_for_vocab=key_for_vocab
        self.model_path=model_path
        self.model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.key_for_vocab)

        #use gpu if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type=='cuda':
            self.model.cuda();

    def fit(self,file_path,checkpoint_path,save_path,tweets=None):
        r"""
   		Trains the model on the given learning data by using the trainer class

        Args:
            file_path (:obj:`str`):
               Path that leads to the training data.
            checkpoint_path (:obj:`str`):
               Path to the directory in which the checkpoints should be saved during the training.
            save_path (:obj:`str`):
               Path where the model should be saved
            tweets (:obj:`Series[str]`):
               if not None: program uses sample data to train the model (only recommended for validation of the function)
    	"""

        if tweets is not None:
            try:
                create_txt_file(tweets=tweets,save_path="./saved_states/temp_file.txt")    
                dataset = LineByLineTextDataset(tokenizer=self.tokenizer, file_path="./saved_states/temp_file.txt",block_size=128)

            finally:
                os.remove("./saved_states/temp_file.txt")

        else:
            dataset = LineByLineTextDataset(tokenizer=self.tokenizer, file_path=file_path,block_size=128)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        self.model.train();

        training_args = TrainingArguments(
    					output_dir=checkpoint_path,
    					overwrite_output_dir=True,
    					num_train_epochs=4,
    					save_steps=10_000,
    					save_total_limit=2,
    					do_train=True,
    					logging_dir='./logs'
						)

        trainer = Trainer(
    			model=self.model,
    			args=training_args,
    			data_collator=data_collator,
    			train_dataset=dataset,
    			prediction_loss_only=True
				)

        trainer.train()

        trainer.save_model(save_path)

    def predict(self,text):
        """
        Prediction of a masked word within a sentence. 
        
            Args:
                text (:obj:`str`):
                    Sentence containing the string [MASK] at one position
        """

        text= "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        masked_index = tokenized_text.index('[MASK]') 

        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        self.model.eval()
        
        # Predict all tokens
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        
        predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
        predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
        
        print(predicted_token)



