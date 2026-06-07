from dataclasses import dataclass

@dataclass
class HBGLConfig:
    label_list: list[str]
    label_levels: list[int]
    sep_token_id: int
    mask_token_id: int

    sep_label_id: int = -1
    
    @property
    def label_count(self):
        return len(self.label_list)
