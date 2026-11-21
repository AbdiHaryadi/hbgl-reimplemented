from pathlib import Path

import huggingface_hub

from pipeline import HBGLPipeline, HBGLPipelineArguments

def make_ignore_patterns(model_name: str, checkpoint_number: int):
    prefix = f"models/{model_name}/ckpt-"
    checkpoint_number_str = str(checkpoint_number)

    index = 0
    ignore_patterns = []
    while index < len(checkpoint_number_str):
        for i in range(10):
            if index == 0 and i == 0:
                continue

            if checkpoint_number_str[index] != str(i):
                ignore_patterns.append(prefix + str(i) + "*")
        prefix += checkpoint_number_str[index]
        index += 1

    return ignore_patterns

def download_wos_hierarchical_pipeline():
    return download_and_load(
        repo_id="abdiharyadi/hbgl-wos",
        model_name="wos",
        checkpoint_number=93000,
        max_seq_length=509,
        max_tgt_length=3,  # Depth maximum + 1
    )

def download_wos_flat_pipeline():
    return download_and_load(
        repo_id="abdiharyadi/hbgl-wos-flat",
        model_name="wos",
        checkpoint_number=96000,
        max_seq_length=509,
        max_tgt_length=3,  # Depth maximum + 1
    )

def download_bgc_hierarchical_pipeline():
    return download_and_load(
        repo_id="abdiharyadi/hbgl-bgc",
        model_name="bgc",
        checkpoint_number=96000,
        max_seq_length=500,
        max_tgt_length=5,  # Depth maximum + 1
    )

def download_bgc_flat_pipeline():
    return download_and_load(
        repo_id="abdiharyadi/hbgl-bgc-flat",
        model_name="bgc",
        checkpoint_number=67500,
        max_seq_length=500,
        max_tgt_length=5,  # Depth maximum + 1
    )

def download_and_load(
        repo_id: str,
        model_name: str,
        checkpoint_number: int,
        max_seq_length: int,
        max_tgt_length: int
):
    local_dir = Path(".")
    huggingface_hub.snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=make_ignore_patterns(model_name, checkpoint_number)
    )
    
    args = HBGLPipelineArguments(
        tokenizer_name="bert-base-uncased",
        max_seq_length=max_seq_length,
        max_tgt_length=max_tgt_length, # Depth maximum + 1
        add_vocab_file=str(local_dir / "label_map.pkl"),
    )
    model_path = str(local_dir / "models" / model_name / f"ckpt-{checkpoint_number}")

    return HBGLPipeline(args=args, model_path=model_path)
