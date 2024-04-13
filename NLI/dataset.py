"""
File for all dataset related functions and classes.
"""

from datasets import load_dataset
from torch.utils.data import Dataset
from nltk import tokenize
import datasets


def load_data() -> datasets.Dataset:
    """
    Loads the three dataset files from hugging face.
    returns: tuple with train, test, validation dataset
    """
    datasets.disable_caching()  # to make sure that tokenizer isn't cached
    data_files = {
        "train": "plain_text/train-00000-of-00001.parquet",
        "test": "plain_text/test-00000-of-00001.parquet",
        "validation": "plain_text/validation-00000-of-00001.parquet",
    }
    dataset = load_dataset("stanfordnlp/snli", data_files=data_files, cache_dir=None)

    return dataset


def preprocess(row) -> dict:
    """
    Applies lowercasing and nltk's implementation of a tokenizer
    """
    text_labels = ["premise", "hypothesis"]
    for label in text_labels:
        row[label] = tokenize.word_tokenize(row[label].lower())

    # pad the sentences to the max length in the whole dataset.
    row["premise"] = row["premise"] + ["<pad>" for i in range(82 - len(row["premise"]))]
    row["hypothesis"] = row["hypothesis"] + [
        "<pad>" for i in range(82 - len(row["hypothesis"]))
    ]

    return row


def preprocess_data(data) -> datasets.Dataset:
    """
    Efficiently preprocesses whole dataset
    """
    data = data.map(preprocess)
    # when there is no consensus amongst the people that labeled the dataset, a label of -1 is assigned
    # this was not shown on Huggingface though :(
    data = data.filter(lambda row: row["label"] != -1)
    return data


class SNLIDataLoader(Dataset):
    """
    Dataloader class for the SNLI dataset.

    """

    def __init__(self, data):
        """
        Initialize dataset class.
        Requires dataset and then extracts the premises, hypothesesisses and labels
        """
        data = preprocess_data(data)
        self.data_premises = data["premise"]
        self.data_hypothesis = data["hypothesis"]
        self.data_label = data["label"]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            self.data_premises[index],
            self.data_hypothesis[index],
            self.data_label[index],
        )
