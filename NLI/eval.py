"""
Eval functions
"""

from utils import process_sentence, process_senteval, create_vocab
import torch
import os
import numpy as np
import pickle
import model
import senteval
import argparse


def eval_model(model, data, device, vocab, loss_function) -> tuple:
    """
    Evaluate a certain model on a certain dataset, calculates accuracy
    """

    correct = 0
    total = 0
    loss = 0
    model.eval()

    for premises, hypotheses, labels in data:
        premises, hypotheses, labels = process_sentence(
            premises, hypotheses, labels, vocab, device
        )

        premises = premises.permute(2, 1, 0)
        hypotheses = hypotheses.permute(2, 1, 0)

        with torch.no_grad():
            logits = model(premises, hypotheses)

        predictions = logits.argmax(dim=-1).view(-1)

        # add the number of correct predictions to the total correct
        correct += (predictions == labels.view(-1)).sum().item()
        total += labels.size(0)

        if loss_function is not None:
            item_loss = loss_function(logits, labels)
            loss += item_loss.item()

    return loss / len(data), (correct / float(total))


def batcher(params, batch):
    batch = [sent if sent != [] else ["."] for sent in batch]
    padded_indices = process_senteval(params.vocab, batch).to(params.device)
    embeddings = params.net.encoder(padded_indices)
    return embeddings.detach().cpu().numpy()


def main(args):
    device = args.device
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    print("First load/build the vocab and embeddings")

    if os.path.isfile("data/" + "embeddings.txt"):
        print("Found required files")
        vectors = np.loadtxt("data/" + "embeddings.txt")
        vectors = torch.Tensor(vectors)

        with open("data/" + "vocab.pickle", "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab, vectors = create_vocab()
        vectors = torch.Tensor(vectors)

    if args.model == "baseline":
        base = model.Baseline(vectors).to(device)
        net = model.NeuralNet(base, 300).to(device)

    if args.model == "LSTM":
        lstm = model.LSTM(vectors, 2048).to(device)
        net = model.NeuralNet(lstm, 2048).to(device)

    if args.model == "BILSTM":
        bi_lstm = model.BiLSTM(vectors, 4096, False).to(device)
        net = model.NeuralNet(bi_lstm, 4096).to(device)

    if args.model == "BILSTM_MAX":
        bi_lstm = model.BiLSTM(vectors, 4096, True).to(device)
        net = model.NeuralNet(bi_lstm, 4096).to(device)

    net.load_state_dict(torch.load(f"models/{args.model}.pt"))
    print(net)
    params = dict()

    params["args"] = args
    params["kfold"] = 5
    params["device"] = args.device
    params["model"] = args.model
    params["net"] = net
    params["task_path"] = "SentEval/data/"
    params["usepytorch"] = True
    params["vocab"] = vocab
    params["batch_size"] = 64

    params["classifier"] = {
        "nhid": 0,
        "optim": "rmsprop",
        "batch_size": 128,
        "tenacity": 3,
        "epoch_size": 2,
    }

    se = senteval.engine.SE(params, batcher, None)
    tasks = [
        # "MR",
        # "CR",
        "MPQA",  # THIS ONE GIVES A PROBLEM
        # "SUBJ",
        # "SST2",  # ALSO GIVES A PROBLEM
        # "TREC",
        # "MRPC",
        # "SICKEntailment",
        # "STS14",  # ALSO GIVES A PROBLEM
        # "SICKRelatedness",  # ALSO GIVES A PROBLEM
    ]
    print("Start eval on senteval")
    outputs = se.eval(tasks)

    tasks_with_devacc = [task for task in tasks if "devacc" in outputs[task]]
    macro_score = np.mean([outputs[task]["devacc"] for task in tasks_with_devacc])
    micro_score = np.sum(
        [outputs[task]["ndev"] * outputs[task]["devacc"] for task in tasks_with_devacc]
    ) / np.sum([outputs[task]["ndev"] for task in tasks_with_devacc])

    print(macro_score)
    print(micro_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "LSTM", "BILSTM", "BILSTM_MAX"],
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="SentEval/data",
        help="Path to the SentEval data directory",
    )

    args = parser.parse_args()

    main(args)
