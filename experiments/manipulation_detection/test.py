from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np


def calculate_metrics(confusion_matrix):
    # Unpack confusion matrix
    TN, FP, FN, TP = confusion_matrix.ravel()

    # Calculate precision, recall, and accuracy
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Calculate F1 scores
    # For binary classification, micro F1 is the same as accuracy
    micro_f1 = accuracy

    # Calculate macro F1-score
    # Macro F1-Score is the average of F1-scores of both classes
    if precision + recall == 0:
        f1_positive = 0
    else:
        f1_positive = 2 * (precision * recall) / (precision + recall)

    precision_negative = TN / (TN + FN) if TN + FN > 0 else 0
    recall_negative = TN / (TN + FP) if TN + FP > 0 else 0
    if precision_negative + recall_negative == 0:
        f1_negative = 0
    else:
        f1_negative = 2 * (precision_negative * recall_negative) / (precision_negative + recall_negative)

    macro_f1 = (f1_positive + f1_negative) / 2
    return {
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy,
        'Micro F1-Score': micro_f1,
        'Macro F1-Score': macro_f1
    }


# Example usage:
conf_matrix = np.array([[8, 176],  # TN, FP
                        [1, 398]])  # FN, TP
metrics = calculate_metrics(conf_matrix)
print(metrics)
