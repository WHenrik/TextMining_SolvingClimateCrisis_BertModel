def extract_classes(data):
	"""
        Receives a pandas DataFrame containing tweets and labels 

        Args:
            data (:obj:`Series[str],Series[str]`):
                Series of tweets and their labels
           
        Returns:
           3 x  :obj:`Series[str],Series[str]`: datasets that are representing their class exclusively
    """

	industry_data=data.text[data.label=="industry"]
	industry_data=industry_data.reset_index(drop=True)

	polit_data=data.text[data.label=="politician"]
	polit_data=polit_data.reset_index(drop=True)

	news_data=data.text[data.label=="news"]
	news_data=news_data.reset_index(drop=True)


	return industry_data, polit_data, news_data

