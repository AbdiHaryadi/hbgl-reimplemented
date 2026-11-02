import json
import logging
import os
from pathlib import Path
import random
import re
from typing import List, Union, Dict, Any

from bs4 import BeautifulSoup
import huggingface_hub
import pandas as pd
from tqdm import tqdm
import wandb

from args import HBGLTrainingArguments
import main
from hierarchy import HIERARCHY_ROOT_NAME, LabelHierarchy
from preprocess import PreprocessorV2 as Preprocessor, PreprocessorConfigV2 as PreprocessorConfig

logging.getLogger().setLevel(logging.INFO)

def get_tag_from_soup_or_raise(soup: BeautifulSoup, name: str):
    el = soup.find(name)
    if el is None:
        raise ValueError(f"{name} is not found")
    return el

def parse_books(file_path: Union[Path, str]) -> List[Dict[str, Any]]:
    """
    Parse a "book" DOM file extracted from the TXT splits.

    :param file_path: training/test/validation file as shared by dataset's authors
    :return list of dictionary with summaries and list of topics
    """

    logging.info(f"Parsing {file_path}...")
    data = list()
    soup = BeautifulSoup(open(file_path, "rt", encoding='utf-8').read(), "html.parser")
    for book in tqdm(soup.find_all("book"), desc="Reading BGC"):
        categories = set()
        per_level_cats = dict()
        book_soup = BeautifulSoup(str(book), "html.parser")
        summary = str(get_tag_from_soup_or_raise(book_soup, "body").string)
        title = str(get_tag_from_soup_or_raise(book_soup, "title").string)
        summary = title + ". " + summary
        for t in book_soup.find_all("topics"):
            s1 = BeautifulSoup(str(t), "html.parser")
            structure = ["d0", "d1", "d2", "d3"]
            for i, level in enumerate(structure):
                l_cat = set()
                for t1 in s1.find_all(level):
                    cat = re.sub(r"\s+", "-", re.sub(r"[^\w\s]", "", str(t1.string).strip())).strip()
                    categories.add(cat)
                    l_cat.add(cat)
                per_level_cats[f"d{i}_topics"] = list(l_cat)
        if summary and categories:
            data.append({"text": summary, "labels": list(categories), "flat_label": "_".join(list(categories)),
                         **per_level_cats})
        else:
            logging.warning("Skipping: no summary or category.")

    random.shuffle(data)
    return data


def write_bgc_jsonl(filepath: Path, data: List):
    with open(filepath, "w") as out_f:
        for s in data:
            out_f.write(f"{json.dumps(s)}\n")

def parse_and_write(input_filepath: Path, output_filepath: Path):
    if os.path.exists(output_filepath):
        print(f"Skip processing {input_filepath} because {output_filepath} already exists.")
        return
    
    data = parse_books(input_filepath)
    data = [{k: v for k, v in d.items() if k not in ["labels", "flat_label"]} for d in data]
    write_bgc_jsonl(output_filepath, data)

def load_df_from_jsonl(path: str | Path):
    rows = []
    with open(path) as fp:
        for line in fp:
            data = json.loads(line)
            rows.append({"text": data["text"]} | {
                f"label_{depth_index}_{label_index}": label
                for depth_index in range(4)
                for label_index, label in enumerate(data[f"d{depth_index}_topics"])
            })
    return pd.DataFrame(rows)

def save_preprocessed_data(data: list[dict[str, Any]], path: str | Path):
    with open(path, mode="w") as fp:
        for instance in data:
            print(json.dumps(instance), file=fp)

def preprocess_procedure(
        input_train_jsonl_path: Path,
        input_test_jsonl_path: Path,
        input_val_jsonl_path: Path,
        input_hierarchy_path: Path,
        output_train_jsonl_path: Path,
        output_test_jsonl_path: Path,
        output_val_jsonl_path: Path,
        output_label_map_path: Path,
        output_hierarchy_path: Path,
):
    train_df = load_df_from_jsonl(input_train_jsonl_path)
    test_df = load_df_from_jsonl(input_test_jsonl_path)
    val_df = load_df_from_jsonl(input_val_jsonl_path)
    assert (
        set(train_df.columns)
        == set(test_df.columns)
        == set(val_df.columns)
    )

    hierarchical_setting = True
    config = PreprocessorConfig(
        text_attribute="text",
        label_attribute_list=(
            [
                [c for c in train_df.columns
                 if c.startswith(f"label_{level_index}")]
                for level_index in range(4)
            ] if hierarchical_setting
            else [[c for c in train_df.columns if c.startswith("label")]]
        )
    )
    preprocessor = Preprocessor(config)
    preprocessed_train_data = preprocessor.fit_transform(train_df)
    preprocessed_test_data = preprocessor.transform(test_df)
    preprocessed_val_data = preprocessor.transform(val_df)

    save_preprocessed_data(preprocessed_train_data, output_train_jsonl_path)
    save_preprocessed_data(preprocessed_test_data, output_test_jsonl_path)
    save_preprocessed_data(preprocessed_val_data, output_val_jsonl_path)

    preprocessor.get_label_mapper().save(output_label_map_path)

    parent_to_childs_map = {}
    with open(input_hierarchy_path) as fp:
        for i, line in enumerate(fp):
            parent, *childs = line.strip().split()
            if i == 0:
                assert parent == "root"
                parent = HIERARCHY_ROOT_NAME
            parent_to_childs_map[parent] = childs
    LabelHierarchy(parent_to_childs_map).save(output_hierarchy_path)

if __name__ == "__main__":
    # Need to run this:
    # wget https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip -O data/blurbgenrecollectionen.zip
    # unzip data/blurbgenrecollectionen.zip -d data/bgc
    # wget https://gitlab.com/distration/dsi-nlp-publib/-/raw/main/htc-survey-24/data/taxonomies/bgc_tax.txt -O data/bgc/bgc_tax.txt
    huggingface_hub.login(os.environ["HUGGINGFACE_TOKEN"])
    wandb.login(os.environ["WANDB_KEY"])

    hf_api = huggingface_hub.HfApi()
    REPO_ID = os.environ["REPO_ID"]
    if REPO_ID:
        hf_api.create_repo(
            repo_id=REPO_ID,
            exist_ok=True,
        )

    data_root = Path("data") / "bgc"

    _train_file = data_root / "BlurbGenreCollection_EN_train.txt"
    _test_file = data_root / "BlurbGenreCollection_EN_test.txt"
    _val_file = data_root / "BlurbGenreCollection_EN_dev.txt"

    _train_file_l = data_root / "BlurbGenreCollection_EN_train.jsonl"
    _test_file_l = data_root / "BlurbGenreCollection_EN_test.jsonl"
    _val_file_l = data_root / "BlurbGenreCollection_EN_dev.jsonl"

    _train_file_prep = data_root / "train_preprocessed.jsonl"
    _test_file_prep = data_root / "test_preprocessed.jsonl"
    _val_file_prep = data_root / "val_preprocessed.jsonl"

    parse_and_write(_train_file, _train_file_l)
    parse_and_write(_test_file, _test_file_l)
    parse_and_write(_val_file, _val_file_l)

    label_map_path = data_root / "label_map.pkl"
    hierarchy_path = data_root / "bgc.taxonomy"

    if not all(os.path.exists(path) for path in [
        _train_file_prep,
        _test_file_prep,
        _val_file_prep,
        label_map_path,
        hierarchy_path,
    ]):
        preprocess_procedure(
            input_train_jsonl_path=_train_file_l,
            input_test_jsonl_path=_test_file_l,
            input_val_jsonl_path=_val_file_l,
            input_hierarchy_path=data_root / "bgc_tax.txt",
            output_train_jsonl_path=_train_file_prep,
            output_test_jsonl_path=_test_file_prep,
            output_val_jsonl_path=_val_file_prep,
            output_label_map_path=label_map_path,
            output_hierarchy_path=hierarchy_path
        )
    else:
        print("Skip preprocess because all outputs are already available")

    args = HBGLTrainingArguments(
        train_file=str(_train_file_prep),
        valid_file=str(_val_file_prep),
        test_file=str(_test_file_prep),
        model_name_or_path="bert-base-uncased",
        output_dir="models/bgc",

        max_source_seq_length=500,
        max_target_seq_length=5,  # Depth maximum + 1, jadi harusnya kalau datar, ya 2.
        per_gpu_train_batch_size=12,
        gradient_accumulation_steps=1,

        add_vocab_file=str(label_map_path),
        label_smoothing=0.0,  # Always use this value
        wandb=True,
        learning_rate=3e-5,
        num_warmup_steps=500,
        num_training_steps=96000,
        cache_dir=".cache",
        seed=120,

        label_cpt=str(hierarchy_path),
        label_cpt_not_incr_mask_ratio=True,
        label_cpt_steps=300,
        label_cpt_use_bce=True,
    )
    try:
        main.main(args)
    finally:
        hf_api.upload_folder(
            folder_path=args.output_dir,
            repo_id=REPO_ID,
            path_in_repo=args.output_dir,
        )
        for path in [
            label_map_path,
            hierarchy_path,
            _train_file_prep,
            _test_file_prep,
            _val_file_prep,
        ]:
            hf_api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path.name,
                repo_id=REPO_ID,
            )
