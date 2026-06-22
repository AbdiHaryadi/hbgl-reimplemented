from dataclasses import asdict, dataclass

@dataclass
class HBGLConfig:
    label_list: list[str]
    label_levels: list[int]
    sep_token_id: int
    mask_token_id: int

    sep_label_id: int = -1
    max_length: int | None = None
    loss_reduction: str = "sum"
    
    @property
    def label_count(self):
        return len(self.label_list)
    
    def to_dict(self):
        return asdict(self)

def child_map_to_label_level_map(child_map: dict[str, list[str]]):
    label_level_map: dict[str, int] = {}

    label_with_childs: list[str] = []
    label_with_parents: set[str] = set()
    for parent, children in child_map.items():
        label_with_childs.append(parent)
        label_with_parents |= set(children)
    
    level = 1
    new_parents: list[str] = []
    for label in label_with_childs:
        if label not in label_with_parents:
            label_level_map[label] = level
            new_parents.append(label)

    stop = False
    while not stop:
        if len(new_parents) == 0:
            stop = True
        else:
            level += 1
            parents = new_parents
            new_parents = []

            for parent in parents:
                if parent in child_map:
                    for child in child_map[parent]:
                        label_level_map[child] = level
                        new_parents.append(child)
    
    return label_level_map

class Hierarchy:
    def __init__(self, data: dict[str, list[str]]) -> None:
        self._data = data

        label_level_map = child_map_to_label_level_map(data)
        self._labels = list(label_level_map)
        self._levels = [label_level_map[label] for label in self._labels]
    
    def get_labels(self):
        return self._labels.copy()
    
    def get_levels(self):
        return self._levels.copy()
    
    def get_all_parent_child_pairs(self):
        results: list[tuple[str, str]] = []
        for parent, childs in self._data.items():
            for c in childs:
                results.append((parent, c))
        
        return results
    
    def has_child_relation(self, label_i: str, label_j: str):
        return label_j in self._data.get(label_i, [])
    
    def is_sibling(self, label_i: str, label_j: str):
        for childs in self._data.values():
            if label_i in childs and label_j in childs:
                return True
            
        return False
    
    def is_leaf(self, label: str):
        return (
            label not in self._data
            or len(self._data[label]) == 0
        )
