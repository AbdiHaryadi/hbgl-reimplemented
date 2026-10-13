from collections import defaultdict
import pickle

from utils import id_to_label_token

SEPARATOR = "\t"
HIERARCHY_ROOT_NAME = "Root"

class LabelHierarchy:
    def __init__(
            self,
            parent_childs_map: dict[str, list[str]],
            # label_to_id_map: dict[str, int],
    ):
        self.parent_childs_map = parent_childs_map
        self.child_parent_map: dict[str, str] = {}
        for parent, child_tokens in parent_childs_map.items():
            for child in child_tokens:
                self.child_parent_map[child] = parent

    def iter_parent_child_name_pairs(self):
        for parent, child_tokens in self.parent_childs_map.items():
            for child in child_tokens:
                yield parent, child

    def get_label_depth(self, label_name: str):
        return len(self.find_path(label_name))

    def find_path(self, label_name: str):
        path: list[str] = []
        while label_name != HIERARCHY_ROOT_NAME:
            path.insert(0, label_name)
            label_name = self.child_parent_map[label_name]
        return path

    def iter_unique_child_names(self):
        visited_names: set[str] = set()
        for _, token in self.iter_parent_child_name_pairs():
            if token not in visited_names:
                yield token
                visited_names.add(token)

    def get_label_index_map(self):
        _label_index_map: dict[str, int] = {}
        for token in self.iter_unique_child_names():
            _label_index_map[token] = len(_label_index_map)
        return _label_index_map

    # def get_label_id(self, label_name: str):
    #     return self.label_to_id_map[label_name] if label_name != ROOT_NAME else -1

    def get_one_by_one_label_init_map(self):
        return {
            label: '/'.join(self.find_path(label))
            for label in self.iter_unique_child_names()
        }

    def save(self, path):
        with open(path, 'w') as f:
            for parent, childs in self.parent_childs_map.items():
                f.write(SEPARATOR.join([parent] + childs) + '\n')

    @classmethod
    def load(cls, parent_childs_path):
        parent_childs_map = defaultdict(list)
        with open(parent_childs_path) as f:
            for line in f.readlines():
                line = line.strip().split(SEPARATOR)
                for token in line[1:]:
                    parent_childs_map[line[0]].append(token)
        return cls(parent_childs_map)

class LabelMapper:
    def __init__(self, label_to_id_map: dict[str, int]):
        self.label_to_id_map = label_to_id_map

    def get_all_labels(self):
        return list(self.label_to_id_map.keys())

    def get_all_ids(self):
        return list(self.label_to_id_map.values())

    def label_to_id(self, label: str):
        # Kinda coupled with LabelHierarchy, but it's okay for now.
        return self.label_to_id_map[label] if label != HIERARCHY_ROOT_NAME else -1

    def label_to_token(self, label: str):
        return id_to_label_token(self.label_to_id(label))

    def get_all_tokens(self):
        return [self.label_to_token(label) for label in self.get_all_labels()]

    def __len__(self):
        return len(self.label_to_id_map)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.label_to_id_map, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return cls(pickle.load(f))
