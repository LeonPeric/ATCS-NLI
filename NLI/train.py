"Training loop file"

import torch
from eval import eval_model
from utils import process_sentence, create_vocab
from tqdm import tqdm
import os
import numpy as np
import pickle
from dataset import SNLIDataLoader, load_data
from torch.utils.data import DataLoader

import model
import torch
import train
import eval

import argparse


def train_model(model, device, vocab, dataloader_train, dataloader_test, model_name):
    """
    The general training loop file. Runs till the learning rate is lower than 10**-5.
    As well as follows all other hyperparameters from the paper.
    Saves the best running model on the test set
    """

    epoch = 0
    test_accuracies = []
    train_loss = 0
    best_eval = 0
    losses = []

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_module = torch.nn.CrossEntropyLoss()

    # from the paper
    while optimizer.param_groups[0]["lr"] >= 10**-5:
        model.train()
        print(epoch)
        if epoch >= 1:
            optimizer.param_groups[0]["lr"] *= 0.99  # from the paper
            print(optimizer.param_groups[0]["lr"])

        for premises, hypotheses, labels in tqdm(
            dataloader_train, total=len(dataloader_train)
        ):
            premises, hypotheses, labels = process_sentence(
                premises, hypotheses, labels, vocab, device
            )
            premises = premises.permute(2, 1, 0)
            hypotheses = hypotheses.permute(2, 1, 0)

            preds = model(premises, hypotheses)
            B = labels.size(0)

            loss = loss_module(preds.view([B, -1]), labels.view(-1))
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Training loss is: {train_loss}")
        losses.append(train_loss)
        train_loss = 0

        test_accuracy = eval_model(model, dataloader_test, device, vocab)[-1]
        print(f"Test accuracy is: {test_accuracy}")

        if len(test_accuracies) == 0:
            test_accuracies.append(test_accuracy)

        # take a fith of the learning rate if test acc does not improve
        if test_accuracy < test_accuracies[-1]:
            optimizer.param_groups[0]["lr"] *= 0.2

        if test_accuracy > best_eval:
            best_eval = test_accuracy
            path = f"models/{model_name}.pt"
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
        test_accuracies.append(test_accuracy)
        epoch += 1


def main(args):
    print(os.getcwd())
    print(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    print("First load/build the vocab and embeddings")
    dataset_path = args.dataset_path

    if os.path.isfile(dataset_path + "embeddings.txt"):
        print("Found required files")
        vectors = np.loadtxt(dataset_path + "embeddings.txt")
        vectors = torch.Tensor(vectors)

        with open(dataset_path + "vocab.pickle", "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab, vectors = create_vocab()
        vectors = torch.Tensor(vectors)

    print("Load dataset")

    datasets = load_data()

    print("Create dataloaders")
    dataloader_train = DataLoader(
        SNLIDataLoader(datasets["train"]), shuffle=True, batch_size=64
    )
    dataloader_test = DataLoader(
        SNLIDataLoader(datasets["test"]), shuffle=True, batch_size=64
    )

    print("Creating model")
    if args.model == "baseline":
        base = model.Baseline(vectors).to(device)
        net = model.NeuralNet(base, 300).to(device)

    if args.model == "LSTM":
        lstm = model.LSTM(vectors).to(device)
        net = model.NeuralNet(lstm, 2048).to(device)

    if args.model == "BILSTM":
        bi_lstm = model.BiLSTM(vectors).to(device)
        net = model.NeuralNet(bi_lstm, True, 4096).to(device)

    if args.model == "BILSTM_MAX":
        bi_lstm = model.BiLSTM(vectors, True).to(device)
        net = model.NeuralNet(bi_lstm, True, 4096).to(device)

    print("The choosen model is: ")
    print(net)

    if args.checkpoint_path:
        print("Found already trained model")
        model.load_state_dict(torch.load(args.checkpoint_path))
    else:
        print("Start training now")
        train.train_model(
            net, device, vocab, dataloader_train, dataloader_test, args.model
        )

    print("Validating now on validation test")
    model.load_state_dict(torch.load(f"models/{args.model}.pt"))
    dataloader_validation = DataLoader(
        SNLIDataLoader(datasets["validation"]), shuffle=True, batch_size=64
    )

    validation_accuracy = eval_model(model, dataloader_validation, device, vocab)[-1]
    print(f"Validation accuracy is: {validation_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/",
        help="Path to embeddings and vocab files",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "LSTM", "BILSTM", "BILSTM_MAX"],
        help="Model type",
    )

    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Trained model path"
    )

    args = parser.parse_args()

    main(args)
