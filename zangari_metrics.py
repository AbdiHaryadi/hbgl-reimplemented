from pathlib import Path
from typing import Dict, List, Union, Optional

import joblib
import networkx as nx
import numpy as np
import torch
from networkx import read_adjlist, Graph, is_tree, shortest_path_length
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, argmax_flag: bool = True, threshold=.5) -> Dict[str, float]:
    """
    Computes regular metrics utilizing sklearn.

    :param threshold: threshold in [0, 1] to consider a prediction positive (> t)
    :param y_true: true labels
    :param y_pred: predicted labels
    :param argmax_flag: Whether to apply an "argmax" function over the predictions
    :return: metrics in a dictionary [metric_name]: value
    """
    # Compute metrics with sklearn
    if argmax_flag:
        y_pred = y_pred.argmax(axis=-1)
    # Thresholding
    if threshold != -1:  # NOTE: NEW: Special case for already transformed labels
        y_pred = np.where(y_pred > threshold, 1, 0).astype(int)
        y_true = np.where(y_true > threshold, 1, 0).astype(int)
    # Compute metrics
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred,
                                                                   average="macro",
                                                                   zero_division="warn")
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics = {
        "accuracy": acc,
        "macro_f1": fscore,
        "macro_precision": precision,
        "macro_recall": recall
    }
    return metrics

def _assign_binarization(labels_2_convert, label_2_index, number_of_classes):
    indicized_labels = [[label_2_index[label] for label in label_list] for label_list in labels_2_convert]
    # multihot_labels = indicized_labels
    multihot_labels = []
    for labels in indicized_labels:
        # manual multihot
        multihot_labels.append(torch.zeros(number_of_classes).scatter(0, torch.Tensor(labels).long(), 1))

    multihot_labels = torch.stack(multihot_labels)
    return multihot_labels


# ************* HIERARCHICAL METRICS *************
def compute_hierarchical_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 taxonomy_path: Path,
                                 encoder_dump_or_mapping: Optional[Union[Path, Dict, MultiLabelBinarizer]],
                                 threshold=.5):
    """
    Compute some hierarchical metrics

    :param y_true: ground truth as MHE labels
    :param y_pred: prediction array with same size as 'y_true', with either probabilities or thresholded values
    :param taxonomy_path: path to file that describes the taxonomy tree
    :param encoder_dump_or_mapping: dump to sklearn MultiLabelBinarizer encoder object
    :param threshold: threshold in [0, 1] to consider a prediction positive (> t)
    :return: dictionary with AHC, and hierarchical metrics, as [metric_name]: value
    """
    # NOTE: for BGC we will need to use read_edgelist to read "blurb_edges.txt"
    g: Graph = read_adjlist(taxonomy_path, nodetype=str)
    assert is_tree(g), "Taxonomy file does not define a Tree, and it should :("
    assert y_pred.shape[0] == y_true.shape[0], f"Different size in pred ({y_pred.shape}) and truth ({y_true.shape})"

    # Convert predictions to str of labels
    if isinstance(encoder_dump_or_mapping, Path) or isinstance(encoder_dump_or_mapping, str):
        lb_encoder: MultiLabelBinarizer = joblib.load(encoder_dump_or_mapping)
        # Duplicated, but ok
        predicted_labels = lb_encoder.inverse_transform(np.where(y_pred > threshold, 1, 0).astype(int))
        true_labels = lb_encoder.inverse_transform(np.where(y_true > threshold, 1, 0).astype(int))
    elif isinstance(encoder_dump_or_mapping, MultiLabelBinarizer):
        predicted_labels = encoder_dump_or_mapping.inverse_transform(
            np.where(y_pred > threshold, 1, 0).astype(int))
        true_labels = encoder_dump_or_mapping.inverse_transform(
            np.where(y_true > threshold, 1, 0).astype(int))
    elif encoder_dump_or_mapping is None:
        # lb_encoder = encoder_dump_or_mapping
        predicted_labels = y_pred
        true_labels = y_true
    else:
        raise NotImplementedError

    ahcs = list()
    for pred_sample, true_labels_sample in zip(predicted_labels, true_labels):
        # AHC
        d = 0.0
        for p in pred_sample:
            d += set_to_label_distance(g, p, true_labels_sample)
        d /= len(true_labels_sample)
        ahcs.append(d)
        # LCA ?

    ahc = np.mean(ahcs)
    h_m = h_multilabel_precision_recall_fscore(true_labels, predicted_labels, taxonomy_path, encoder_dump_or_mapping)
    h_m["ahc"] = float(ahc)
    return h_m


def label_distance(g: Graph, label1: str, label2: str):
    return shortest_path_length(g, source=label1, target=label2)


def set_to_label_distance(g: Graph, label: str, label_set: List[str]):
    # Return dict {target: dist} for each node in g
    return min([label_distance(g, label, k) for k in label_set])


def _extend_label_set(g: nx.DiGraph, labels: List[List[str]]) -> List[List[str]]:
    """
    Extends the label set with all ancestors of each label

    :param g: label taxonomy tree
    :param labels: set of labels to expand
    :return: the new expanded labels
    """
    labels_extended = list()
    for original_sample_labels in labels:
        extended_sample_labels = set(original_sample_labels)
        for label in original_sample_labels:
            a = nx.ancestors(G=g, source=label)
            extended_sample_labels.update(a)
        if "root" in extended_sample_labels:
            # This is a dummy node, and it's not a real label
            extended_sample_labels.remove("root")
        if "Root" in extended_sample_labels:
            extended_sample_labels.remove("Root")
        labels_extended.append(list(extended_sample_labels))
    return labels_extended


def h_multilabel_precision_recall_fscore(labels_true, labels_pred,
                                         taxonomy_path: Path, encoder_dump_or_mapping):
    """
    Compute hierarchical precision, recall and f1-score, as documented in https://arxiv.org/abs/2206.08653

    :param labels_true: ground truth labels (>1) for each sample, as list of strings per sample
    :param labels_pred: predicted labels for each sample, as list of strings
    :param taxonomy_path: path to file with taxonomy
    :param encoder_dump_or_mapping: dump to MultiLabelBinarizer / mapping to i2v, already fitted on training set
    :return: dictionary of H-metrics
    """
    lb_encoder: MultiLabelBinarizer | None = None
    if isinstance(encoder_dump_or_mapping, Path) or isinstance(encoder_dump_or_mapping, str):
        lb_encoder = joblib.load(encoder_dump_or_mapping)
    elif isinstance(encoder_dump_or_mapping, MultiLabelBinarizer):
        lb_encoder = encoder_dump_or_mapping
    g: nx.DiGraph = read_adjlist(taxonomy_path, nodetype=str, create_using=nx.DiGraph)

    labels_pred_extended: List[List[str]] = _extend_label_set(g, labels_pred)
    labels_true_extended: List[List[str]] = _extend_label_set(g, labels_true)
    if lb_encoder is not None:
        mhe_extended_predictions = lb_encoder.transform(labels_pred_extended)
        mhe_extended_truth = lb_encoder.transform(labels_true_extended)
    else:
        mapping = encoder_dump_or_mapping.v2i['label']  # type: ignore
        mhe_extended_predictions = _assign_binarization(labels_pred_extended, mapping,
                                                                    len(g) - 1)  # -1 cause root
        mhe_extended_truth = _assign_binarization(labels_true_extended, mapping,
                                                              len(g) - 1)  # -1 cause root

    # note: this more damage than it helps
    # assert mhe_extended_predictions.shape == mhe_extended_truth.shape, \
    #     f"Different size in ext pred ({mhe_extended_predictions.shape}) and truth ({mhe_extended_truth.shape})"
    precision, recall, f_score, _ = precision_recall_fscore_support(y_true=mhe_extended_truth,
                                                                    y_pred=mhe_extended_predictions, average="macro",
                                                                    zero_division="warn")
    return dict(h_precision=float(precision), h_recall=float(recall), h_fscore=float(f_score))
