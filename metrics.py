from pathlib import Path

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from zangari_metrics import compute_metrics, compute_hierarchical_metrics

def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f

def evaluate(epoch_predicts, epoch_labels, id2label, tax_file: str | None = None):
    """
    :param epoch_predicts: List[List[int]], predicted list, label id (the previous one support probabilities, but it's unused right now)
    :param epoch_labels: List[List[int]], ground truth, label id
    :param id2label: dict[int, str]
    :return: Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float, 'full' -> dict}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        sample_predict_id_list = sample_predict
        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    original_result = {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}

    if tax_file is None:
        return original_result

    return original_result | zangari_evaluate(epoch_predicts, epoch_labels, id2label, tax_file)


def zangari_evaluate(epoch_predicts, epoch_labels, id2label, tax_file):
    # Transform labels into required format for hierarchical metrics (samples, labels)
    multi_label_binarizer = MultiLabelBinarizer()
    y_pred = [[id2label[label] for label in labels] for labels in epoch_predicts]
    y_true = [[id2label[label] for label in labels] for labels in epoch_labels]
    y_true_bin = multi_label_binarizer.fit_transform(y_true)
    y_pred_bin = multi_label_binarizer.transform(y_pred)
    assert isinstance(y_true_bin, np.ndarray)
    assert isinstance(y_pred_bin, np.ndarray)

    metrics = compute_metrics(y_true_bin, y_pred_bin, argmax_flag=False, threshold=-1)
    if tax_file is not None:
        h_metrics = compute_hierarchical_metrics(y_true_bin, y_pred_bin, encoder_dump_or_mapping=multi_label_binarizer,
                                             taxonomy_path=Path(tax_file))
    else:
        h_metrics = {}

    metrics = {f"{k}_zangari": v for k, v in metrics.items()}

    extended_result = metrics | h_metrics
    return extended_result
