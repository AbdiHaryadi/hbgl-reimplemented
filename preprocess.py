from dataclasses import dataclass
import re
from typing import TypeVar

import pandas as pd

from hierarchy import LabelHierarchy, LabelMapper


def wos_text_transform(string: str):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    You should use it before using Preprocessor for WOS data.
    """
    string = string.strip().strip('"')
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class PreprocessorConfig:
    def __init__(
            self,
            text_attribute: str,
            label_attribute_list: str | list[str] | list[list[str]],
            label_id_attribute_list = None,
    ):
        label_attribute_list = two_dimension_processing(label_attribute_list)
        if isinstance(label_attribute_list, str):
            label_attribute_list = [[label_attribute_list]]
        elif isinstance(label_attribute_list[0], str):
            new_label_attribute_list: list[list[str]] = [[]]
            for label in label_attribute_list:
                assert isinstance(label, str)
                new_label_attribute_list[0].append(label)
            label_attribute_list = new_label_attribute_list
        else:
            new_label_attribute_list: list[list[str]] = []
            for label in label_attribute_list:
                assert isinstance(label, list)
                new_label_attribute_list.append(label)
            label_attribute_list = new_label_attribute_list

        if label_id_attribute_list is None:
            label_id_attribute_list = label_attribute_list
        else:
            label_id_attribute_list = two_dimension_processing(label_id_attribute_list)

        self.text_attribute = text_attribute
        self.label_attribute_list = label_attribute_list
        self.id_attribute_list = label_id_attribute_list

T = TypeVar('T')
def two_dimension_processing(x: T | list[T] | list[list[T]]) -> list[list[T]]:
    if not isinstance(x, list):
        return [[x]]
    
    if not isinstance(x[0], list):
        new_x: list[list[T]] = [[]]
        for xi in x:
            assert not isinstance(xi, list)
            new_x[0].append(xi)  # type: ignore
        return new_x
    else:
        new_x: list[list[T]] = []
        for xi in x:
            assert isinstance(xi, list)
            new_x.append(xi)
        return new_x

def generate_code_to_most_frequent_label_map(df: pd.DataFrame, config: PreprocessorConfig) -> dict[tuple[int, str], str]:
    stats: dict[tuple[int, str], dict[str, int]] = {}
    for _, row in df.iterrows():
        for label_attribute, id_attribute in zip(config.label_attribute_list, config.id_attribute_list):
            for level_index, (label_attribute_i, id_attribute_i) in enumerate(zip(label_attribute, id_attribute)):
                if pd.isna(row[label_attribute_i]) or pd.isna(row[id_attribute_i]):
                    break

                key = (level_index, str(row[id_attribute_i]).strip())
                label = row[label_attribute_i].strip()
                if key not in stats:
                    stats[key] = {row[label_attribute_i].strip(): 1}
                elif label not in stats[key]:
                    stats[key][label] = 1
                else:
                    stats[key][label] += 1

    return {
        key: max(stats_per_key.keys(), key=lambda x: stats_per_key[x])
        for key, stats_per_key in stats.items()
    }

def get_hierarchy_from_list(data):
    label_hierarchy: dict[str, list[str]] = {}
    for instance in data:
        labels = instance["labels"]

        for label_list in labels:
            current_parent = "Root"
            for label in label_list:
                if current_parent not in label_hierarchy:
                    label_hierarchy[current_parent] = []
                elif label not in label_hierarchy[current_parent]:
                    label_hierarchy[current_parent].append(label)
                current_parent = label
    return label_hierarchy

LabelToIdType = dict[str, int]
def make_label_to_id_map(data: list[dict]) -> LabelToIdType:
    return {label: i for i, label in enumerate(
        sorted(list(set(sum([sum([k for k in j['labels']], []) for j in data], []))))
    )}

def label_to_tgt_list(label_to_id_map: LabelToIdType, labels: list[list[str]]):
    max_depth = max(len(label_list) for label_list in labels)
    return [sorted(list(set([
        label_to_id_map[label_list[level_index]] for label_list in labels if level_index < len(label_list)
    ]))) for level_index in range(max_depth)]

def transform_to_src_tgt(label_to_id_map: LabelToIdType, data: list[dict]):
    return [
        {'src': l['text'], 'tgt': label_to_tgt_list(label_to_id_map, l['labels']) }
        for l in data
    ]

@dataclass
class PreprocessorState:
    code_to_label_map: dict[tuple[int, str], str]
    parent_childs_map: dict[str, list[str]]
    label_to_id_map: LabelToIdType

class Preprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.state: PreprocessorState | None = None

    def fit(self, df: pd.DataFrame):
        code_to_label_map = generate_code_to_most_frequent_label_map(df, config=self.config)
        data = self.data_df_to_data_list(df=df, code_to_label_map=code_to_label_map)
        self.state = PreprocessorState(
            code_to_label_map=code_to_label_map,
            parent_childs_map=get_hierarchy_from_list(data),
            label_to_id_map=make_label_to_id_map(data),
        )
        return self

    def data_df_to_data_list(
            self,
            df: pd.DataFrame,
            code_to_label_map: dict[tuple[int, str], str],
    ):
        data: list[dict] = []
        for _, row in df.iterrows():
            labels = []
            for id_attribute in self.config.id_attribute_list:
                label_list = []
                for level_index, id_attribute_i in enumerate(id_attribute):
                    if pd.isna(row[id_attribute_i]):
                        break
    
                    label_list.append(code_to_label_map[(
                        level_index,
                        str(row[id_attribute_i]).strip()
                    )])
                labels.append(label_list)
            
            data.append({
                'text': row[self.config.text_attribute].strip().lower(),
                'labels': labels
            })
    
        return data

    def transform(self, df: pd.DataFrame):
        assert self.state is not None, "Fit it first."
        data = self.data_df_to_data_list(df=df, code_to_label_map=self.state.code_to_label_map)
        return transform_to_src_tgt(label_to_id_map=self.state.label_to_id_map, data=data)

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def get_hierarchy(self):
        assert self.state is not None, "Fit it first."
        return LabelHierarchy(self.state.parent_childs_map)

    def get_label_mapper(self):
        assert self.state is not None, "Fit it first."
        return LabelMapper(self.state.label_to_id_map)
