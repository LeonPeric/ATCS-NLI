"""
File with functions that are used throughout the project.
"""

from collections import Counter, OrderedDict
from dataset import load_data
from nltk import tokenize
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence


class OrderedCounter(Counter, OrderedDict):
    """
    Counter that remembers the order elements are first seen
    Copied from NLP1 course
    """

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """
    A vocabulary, assigns IDs to tokens
    Copied from NLP1 course
    """

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        """
        Increments value for certain token
        """
        self.freqs[t] += 1

    def add_token(self, t):
        """
        Adds another token to the vocab
        """
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self):
        """
        Builds the vocab while also adding unk and pad
        """

        self.add_token("<unk>")  # reserve 0 for <unk>
        self.add_token("<pad>")  # reserve 1 for <pad>

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            self.add_token(tok)


def create_vocab():
    """
    Generates the whole vocab
    Also partly copied from NLP1 course
    """
    vocab = set()
    print("Loading the datasets")
    dataset = load_data()
    splits = ["train"]
    text_labels = ["premise", "hypothesis"]

    print("Generating vocab set")
    for split in splits:
        data = dataset[split]
        for row in data:
            for label in text_labels:
                vocab.update(tokenize.word_tokenize(row[label].lower()))

    v = Vocabulary()
    features = {}
    total_lines = 2196019
    print("reading embeddings")
    with open("data\\glove.840B.300d.txt", encoding="utf-8") as f:
        for _, line in tqdm(enumerate(f), total=total_lines):
            elements = line.split(" ")
            token = elements[0]
            if token not in vocab:
                continue
            v.count_token(token)
            features[token] = list(map(float, elements[1:]))

    print("building vocab")
    v.build()
    len_feature = len(features[v.i2w[3]])
    vectors = [None] * len(v.w2i)

    print("converting embeddings")

    # build vecs, set vector to zeros for token not in features
    for token in v.i2w:
        if token not in features:
            vectors[v.w2i[token]] = [0] * len_feature
        else:
            vectors[v.w2i[token]] = features[token]

    vectors = np.stack(vectors, axis=0)
    print(f"Matrix Shape: {vectors.shape}")
    np.savetxt("data/embeddings.txt", vectors)

    with open("data/vocab.pickle", "wb") as f:
        pickle.dump(v, f)

    return v, vectors


def process_sentence(premises, hypotheses, labels, vocab, device):
    """
    Map tokens to their IDs for a single example
    # modified from NLP1 course
    """

    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    premises = [[vocab.w2i.get(t, 0) for t in premise] for premise in premises]
    premises = torch.LongTensor([premises]).to(device)

    hypotheses = [
        [vocab.w2i.get(t, 0) for t in hypothesis] for hypothesis in hypotheses
    ]
    hypotheses = torch.LongTensor([hypotheses]).to(device)

    # y = [label for label in labels]
    y = torch.LongTensor(labels)
    y = y.to(device)

    return premises, hypotheses, y


def process_senteval(vocab, sentences):
    sentences = [tokenize.word_tokenize(" ".join(row).lower()) for row in sentences]

    max_length = max([len(sentence) for sentence in sentences])
    # padded_sentences = []
    # for i in range(len(sentences)):
    #     sentences[i] + ["<pad>" for i in range(max_length - len(sentences[i]))]
    tokenized = []
    for sent in sentences:
        temp = torch.tensor([vocab.w2i.get(token, 0) for token in sent])
        tokenized.append(temp)

    padded = pad_sequence(tokenized, batch_first=True, padding_value=1)
    # tokenized = torch.LongTensor([tokenized])
    return padded
