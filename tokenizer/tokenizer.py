from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

def build_tokenizer(data_path, save_path):	
	r"""
        Creates a tokenizer for the Bert Model based on the given data corpus

        Args:
            data_path (:obj:`str`):
            	Path to the data corpus
            save_path (:obj:`str`):
				Path where the custom tokenizer should be saved
    """

    # Initialize a tokenizer
	tokenizer = ByteLevelBPETokenizer()
	# Customize training
	tokenizer.train(files=data_path, vocab_size=52000, min_frequency=2, special_tokens=[
	    "<s>",
	    "<pad>",
	    "</s>",
	    "<unk>",
	    "<mask>",
	])
	tokenizer.save(save_path)

def test_tokenizer(test_sentence,vocab_path,merge_path):
	r"""
        Illustrates how the individual Tokenizer works

        Args:
            test_sentence (:obj:`str`):
            	Sentence for demonstration purposes
            vocab_path (:obj:`str`):
				Path where the vocabulary (most frequent tokens ranked by frequency) is saved
			merge_path (:obj:`str`):
				Path where the merges file is saved
    """

	tokenizer = ByteLevelBPETokenizer(vocab_path, merge_path)

	tokenizer._tokenizer.post_processor = BertProcessing(("</s>", tokenizer.token_to_id("</s>")),("<s>", tokenizer.token_to_id("<s>")))
	tokenizer.enable_truncation(max_length=512)

	print("Original sentence " + test_sentence)
	print("Encoded string: {}".format(tokenizer.encode(test_sentence).tokens))

	encoding = tokenizer.encode(test_sentence)
	decoded = tokenizer.decode(encoding.ids)
	print("Decoded string: {}".format(decoded))


