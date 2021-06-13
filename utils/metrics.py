import numpy as np
from sklearn.metrics import accuracy_score


def calc_accuracy_from_logits(outputs, true_labels, model):
    logits = outputs.cpu().numpy()
    scores = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    predictions = [{"label": item.argmax(), "score": item.max().item()} for item in scores]
    accuracy = accuracy_score(y_true=true_labels.cpu(), y_pred=[item['label'] for item in predictions])
    return accuracy, predictions
