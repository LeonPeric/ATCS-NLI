"Training loop file"

import torch
from tqdm import tqdm
import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import SNLIDataLoader, load_data
import model
import torch
import train
import eval
import utils


def train_model(
    model, device, vocab, dataloader_train, dataloader_validation, model_name
):
    """
    The general training loop file. Runs till the learning rate is lower than 10**-5.
    As well as follows all other hyperparameters from the paper.
    Saves the best running model on the validation set
    """

    epoch = 0
    validation_accuracies = []
    train_loss = 0
    best_eval = 0
    losses = []

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_module = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(f"./runs/{model_name}")

    # from the paper
    while optimizer.param_groups[0]["lr"] >= 10**-5:
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch)
        model.train()
        print(epoch)
        if epoch >= 1:
            optimizer.param_groups[0]["lr"] *= 0.99  # from the paper
            print(optimizer.param_groups[0]["lr"])

        for premises, hypotheses, labels in tqdm(
            dataloader_train, total=len(dataloader_train)
        ):
            premises, hypotheses, labels = utils.process_sentence(
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
        writer.add_scalar("Loss/training", train_loss / len(dataloader_train), epoch)
        losses.append(train_loss)
        train_loss = 0

        validation_loss, validation_accuracy = eval.eval_model(
            model, dataloader_validation, device, vocab, loss_module
        )
        writer.add_scalar("Loss/validation", validation_loss, epoch)
        writer.add_scalar("Loss/difference", train_loss - validation_loss, epoch)
        print(f"validation accuracy is: {validation_accuracy}")
        writer.add_scalar("Validation accuracy", validation_accuracy, epoch)

        if len(validation_accuracies) == 0:
            validation_accuracies.append(validation_accuracy)

        # take a fith of the learning rate if validation acc does not improve
        if validation_accuracy < validation_accuracies[-1]:
            optimizer.param_groups[0]["lr"] *= 0.2

        if validation_accuracy > best_eval:
            best_eval = validation_accuracy
            path = f"models/{model_name}.pt"
            torch.save(model.state_dict(), path)
        validation_accuracies.append(validation_accuracy)
        epoch += 1

    writer.flush()
    writer.close()


def main(args):
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
        vocab, vectors = utils.create_vocab()
        vectors = torch.Tensor(vectors)

    print("Load dataset")

    datasets = load_data()

    print("Create dataloaders")
    dataloader_train = DataLoader(
        SNLIDataLoader(datasets["train"]), shuffle=True, batch_size=64, pin_memory=False
    )
    dataloader_validation = DataLoader(
        SNLIDataLoader(datasets["validation"]),
        shuffle=True,
        batch_size=64,
        pin_memory=False,
    )

    print("Creating model")
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

    print("The choosen model is: ")
    print(net)

    if args.checkpoint_path:
        print("Found already trained model")
        net.load_state_dict(torch.load(args.checkpoint_path))
    else:
        print("Start training now")
        train.train_model(
            net, device, vocab, dataloader_train, dataloader_validation, args.model
        )

    print("Testing now on test dataset")
    net.load_state_dict(torch.load(f"models/{args.model}.pt"))
    dataloader_test = DataLoader(
        SNLIDataLoader(datasets["test"]),
        shuffle=True,
        batch_size=64,
        pin_memory=False,
    )

    test_accuracy = eval.eval_model(
        net, dataloader_test, device, vocab, loss_function=None
    )[-1]
    print(f"Test accuracy is: {test_accuracy}")


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
