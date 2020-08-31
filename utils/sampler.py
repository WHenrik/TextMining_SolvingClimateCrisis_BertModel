from collections import Counter
import pandas as pd



def shrink_major_class_down(data,number_to_be_shrinked=10000):
    r"""
    Receives a pandas DataFrame containing tweets and labels and samples down the major class to the given value

        Args:
            data (:obj:`Series[str],Series[str]`):
                Series of tweets and their labels
            number_to_be_shrinked (:obj:`int`):
                Number to which the major class is to be undersampled
           
        Returns:
            :obj:`DataFrame[Series[str],Series[str]]`: resampled pandas DataFrame containing tweets and labels
    """

    counts=Counter(data.label)
    max_key=max(counts, key=counts.get)

    data_class_major= data[data['label'] == max_key]
    data_class_rest = data[data['label'] != max_key]

    data_class_major_under = data_class_major.sample(10000, random_state=42)
    data_prep = pd.concat([data_class_major_under, data_class_rest], axis=0)
    data_prep=data_prep.reset_index(drop=True)

    return data_prep

def oversample(data):
    """
        Receives a pandas DataFrame containing tweets and labels and oversamples the minor classes to the size of the major class

        Args:
            data (:obj:`Series[str],Series[str]`):
                Series of tweets and their labels
           
        Returns:
            :obj:`DataFrame[Series[str],Series[str]]`: resampled pandas DataFrame containing tweets and labels
    """

    counts=Counter(data.label)
    max_value = max(counts.values())

    # Divide by class
    data_class_0 = data[data['label'] == 0]
    data_class_1 = data[data['label'] == 1]
    data_class_2 = data[data['label'] == 2]
    data_class_3 = data[data['label'] == 3]

    data_class_0_under = data_class_0.sample(max_value, random_state=42)
    data_class_1_over = data_class_1.sample(max_value,replace=True, random_state=42)
    data_class_2_over = data_class_2.sample(max_value,replace=True, random_state=42)
    data_class_3_over = data_class_3.sample(max_value,replace=True, random_state=42)

    data = pd.concat([data_class_0_under, data_class_1_over,data_class_2_over,data_class_3_over], axis=0)

    return data


