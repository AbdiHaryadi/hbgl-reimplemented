from dataclasses import dataclass

@dataclass
class HBGLConfig:
    label_list: list[str]
    label_levels: list[int]
    sep_token_id: int
    mask_token_id: int

    sep_label_id: int = -1
    max_length: int | None = None
    
    @property
    def label_count(self):
        return len(self.label_list)

class Hierarchy:
    def __init__(self, label_level_map: dict[str, int]) -> None:
        self._labels = list(label_level_map)
        self._levels = [label_level_map[label] for label in self._labels]
    
    def get_labels(self):
        return self._labels.copy()
    
    def get_levels(self):
        return self._levels.copy()
