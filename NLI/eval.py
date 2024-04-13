"""
Eval functions
"""

from utils import process_sentence
import torch


def eval_model(model, data, device, vocab) -> tuple:
    """
    Evaluate a certain model on a certain dataset, calculates accuracy
    """

    correct = 0
    total = 0
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

    return correct, total, correct / float(total)
