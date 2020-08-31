def create_txt_file(tweets,save_path):
    """
        Receives a series of tweets and saves it under the given path

        Args:
            tweets (:obj:`Series[str]`):
                Series of tweets
            save_path (:obj:`str`):
                path where the txt-file should be saved
    """

    outF = open(save_path, "w")
    for line in tweets:
        # write line to output file
        outF.write(line)
        outF.write("\n")
    outF.close()
    