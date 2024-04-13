"Training loop file"

import torch
from eval import eval_model
from utils import process_sentence
from tqdm.notebook import tqdm


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
            path = f"../models/{model_name}.pt"
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
        test_accuracies.append(test_accuracy)
        epoch += 1
