"File for all the models implemented in PyTorch"

from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NeuralNet(nn.Module):
    """
    Neural net, which takes an encoder and passes the outputs through Linear layers
    """

    def __init__(self, encoder, input_dim=4096):
        super(NeuralNet, self).__init__()
        self.encoder = encoder
        self.layers = nn.Sequential(nn.Linear(input_dim * 4, 512), nn.Linear(512, 3))

    def forward(self, premise, hypothesis):
        """
        Run model with specific premises and hypothesis
        """
        premise_output = self.encoder(premise)
        hypothesis_output = self.encoder(hypothesis)
        feature_vector = torch.cat(
            (
                premise_output,
                hypothesis_output,
                torch.abs(premise_output - hypothesis_output),
                premise_output * hypothesis_output,
            ),
            dim=1,
        )

        return self.layers(feature_vector)


class Baseline(nn.Module):
    """
    Baseline model which takes an average over the embeddings
    """

    def __init__(self, embedding_matrix):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

    def forward(self, token_ids):
        """
        Run model with specific premises and hypothesis
        """
        token_ids = torch.squeeze(token_ids)
        return self.embedding(token_ids).mean(1)


class LSTM(nn.Module):
    """
    Regular LSTM which is LTR and contains one layer. Returns value of last hidden layer
    """

    def __init__(self, embedding_matrix, hidden_size):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, token_ids):
        """
        Run model with specific premises and hypothesis
        """
        token_ids = torch.squeeze(token_ids)

        # for padding the embeddings
        lengths = torch.tensor([token_ids.shape[-1] for i in range(token_ids.shape[0])])
        counts = (token_ids == 1).sum(dim=1).cpu()
        seq_lengths = lengths - counts

        embeddings = self.embedding(token_ids)

        packed_embeddings = pack_padded_sequence(
            embeddings, seq_lengths, batch_first=True, enforce_sorted=False
        )

        _, (ht, _) = self.lstm(packed_embeddings)

        return ht[-1]


class BiLSTM(nn.Module):
    """
    BiDirectional LSTM which is able to return:
        the last hidden states from LTR and RTL.
        As well as maxpooling the embeddings
    """

    def __init__(self, embedding_matrix, hidden_size=4096, max_pooling=False):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

        # TODO: should the embedding be 4096 * 2 or / 2 since we take a concat?
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.max_pooling = max_pooling

    def forward(self, token_ids):
        token_ids = torch.squeeze(token_ids)

        lengths = torch.tensor([token_ids.shape[-1] for i in range(token_ids.shape[0])])
        counts = (token_ids == 1).sum(dim=1).cpu()
        seq_lengths = lengths - counts

        embeddings = self.embedding(token_ids)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, seq_lengths, batch_first=True, enforce_sorted=False
        )

        outputs, (ht, _) = self.lstm(packed_embeddings)

        if self.max_pooling:
            # unpack the results
            output_unpacked, _ = pad_packed_sequence(outputs, batch_first=True)
            output, _ = torch.max(output_unpacked, dim=1)
            return output

        else:
            ht_forward = ht[-2]
            ht_backward = ht[-1]
            ht = torch.cat((ht_forward, ht_backward), dim=1)
            return ht
