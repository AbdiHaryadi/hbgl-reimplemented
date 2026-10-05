from collections import defaultdict

def find_path(reversed_hiera: dict[str, str], token: str):
    path: list[str] = []
    while token != "Root":
        path.insert(0, token)
        token = reversed_hiera[token]
    return path

class LabelHierarchy:
    def __init__(
            self,
            parent_childs_map: dict[str, list[str]],
            label_to_id_map: dict[str, int],
    ):
        self.parent_childs_map = parent_childs_map
        self.child_parent_map: dict[str, str] = {}
        for parent, child_tokens in parent_childs_map.items():
            for child in child_tokens:
                self.child_parent_map[child] = parent
        self.label_to_id_map = label_to_id_map

    def get_num_hiers(self) -> dict[int, list[int]]:
        num_hiers = defaultdict(list)
        for parent, child in self.iter_hiera_parent_child_name_pairs():
            num_hiers[self.get_label_id(parent) + 1].append(self.get_label_id(child) + 1)
        return num_hiers

    def get_reversed_num_hiers(self) -> dict[int, int]:
        reversed_num_hiers: dict[int, int] = {}
        for parent, child in self.iter_hiera_parent_child_name_pairs():
            reversed_num_hiers[self.get_label_id(child) + 1] = self.get_label_id(parent) + 1
        return reversed_num_hiers

    def iter_hiera_parent_child_name_pairs(self):
        for parent, child_tokens in self.parent_childs_map.items():
            for child in child_tokens:
                yield parent, child

    def iter_hiera_parent_child_id_pairs(self):
        for parent, child in self.iter_hiera_parent_child_name_pairs():
            yield self.get_label_id(parent), self.get_label_id(child)

    def get_label_depth(self, label_name: str):
        return len(self.find_path(label_name))

    def find_path(self, label_name: str):
        path: list[str] = []
        while label_name != "Root":
            path.insert(0, label_name)
            label_name = self.child_parent_map[label_name]
        return path

    def get_label_index_map(self):
        _label_index_map = {}
        for _, token in self.iter_hiera_parent_child_name_pairs():
            if token not in _label_index_map:
                _label_index_map[token] = len(_label_index_map)
        return _label_index_map

    def get_label_id(self, label_name: str):
        return self.label_to_id_map[label_name] if label_name != "Root" else -1

    @classmethod
    def load(cls, hierarchy_path, label_to_id_map: dict[str, int]):
        parent_childs_map = defaultdict(list)
        with open(hierarchy_path) as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                for token in line[1:]:
                    parent_childs_map[line[0]].append(token)
        return cls(parent_childs_map, label_to_id_map)
