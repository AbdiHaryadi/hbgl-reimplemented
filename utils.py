import array
from collections import defaultdict
import json
import logging
from multiprocessing import Pool
import os
import random
import time

import lmdb
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

class TrainingExample(object):
    def __init__(self, source_ids: list[int], target_ids: list[list[int]], example_id):
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.example_id = example_id

Seq2seqDatasetForBertElementType = tuple[list[int], torch.Tensor, torch.Tensor, list[int], int, int]
class Seq2seqDatasetForBert(torch.utils.data.Dataset):
    def __init__(
            self,
            features: list[TrainingExample],
            max_source_len: int,
            max_target_len: int,
            vocab_size: int,
            tokenizer: BertTokenizer,
            offset: int,
            num_training_instances: int, 
            mask_way: str = 'v1',
            target_mask_prob: float = -1.0,
            num_max_mask_token: int = 0 , 
            source_mask_prob: float = -1.0,
            soft_label: bool = False,
    ):
        assert isinstance(tokenizer.cls_token_id, int)
        assert isinstance(tokenizer.sep_token_id, int)
        assert isinstance(tokenizer.pad_token_id, int)
        assert isinstance(tokenizer.mask_token_id, int)
        
        self.features = features
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.offset = offset
        if offset > 0:
            logger.info("  ****  Set offset %d in Seq2seqDatasetForBert ****  ", offset)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.random_prob = 0.0
        self.keep_prob = 0.0
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = vocab_size
        self.num_training_instances = num_training_instances
        self.target_mask_prob = target_mask_prob
        if mask_way == 'v0':
            num_max_mask_token = self.max_target_len
            logger.info("Mask way v0: set num_max_mask_token = %d" % num_max_mask_token)
        self.num_max_mask_token = num_max_mask_token
        self.mask_way = mask_way
        assert mask_way in ('v0', 'v1', 'v2')
        self.source_mask_prob = source_mask_prob

        self.soft_label = soft_label

    def __len__(self):
        return self.num_training_instances

    def __trunk(self, ids: list[int], max_len: int, append_sep=True):
        if append_sep:
            max_len -= 1
        if len(ids) > max_len:
            ids = ids[:max_len]
        if append_sep:
            ids = ids + [self.sep_id]
        return ids

    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids

    def get_masked_token(self):
        # We ignore the case when using random token prob and keep prob
        # since it's not supported in HBGL.
        return self.mask_id

    def soft_label_getitem(self, _idx: int):
        assert self.mask_way == 'v2'
        idx = (self.offset + _idx) % len(self.features)
        # print("%d get %d" % (_idx, idx))
        feature = self.features[idx]
        source_ids = self.__trunk([self.cls_id] + feature.source_ids, self.max_source_len, append_sep=self.mask_way != 'v0')
        target_ids = feature.target_ids
        sep_token_id = sum([len(i) > 0  for i in target_ids])

        target_ids = torch.zeros(len(feature.target_ids) + 1, self.vocab_size)
        for i, ti in enumerate(feature.target_ids):
            for j, tij in enumerate(ti):
                target_ids[i][tij] = 1

        target_ids[sep_token_id, self.sep_id] = 1
        target_ids = target_ids[:sep_token_id+1]

        num_source_tokens = len(source_ids)
        num_target_tokens = self.get_num_target_tokens_by_target_ids(target_ids)


        if self.source_mask_prob > 0:
            for i in range(num_source_tokens):
                tk_id = source_ids[i]
                if tk_id != self.cls_id and tk_id != self.sep_id:
                    r = random.random()
                    if r < self.source_mask_prob:
                        source_ids[i] = self.get_masked_token()

        source_ids = self.__pad(source_ids, self.max_source_len)
        # target_ids = self.__pad(target_ids, self.max_target_len)

        target_ids, label_ids, pseudo_ids = self.unnamed_method(target_ids)
        return (
            source_ids,
            target_ids,
            label_ids,
            pseudo_ids,
            num_source_tokens,
            num_target_tokens,
        )

    def get_num_target_tokens_by_target_ids(self, target_ids: torch.Tensor):
        return target_ids.shape[0]

    def unnamed_method(self, target_ids: torch.Tensor):
        # target_ids is two dimensional
        num_target_tokens = self.get_num_target_tokens_by_target_ids(target_ids)

        pseudo_ids = []
        label_ids = []
        for pos in range(num_target_tokens):
            tk_id = target_ids[pos]
            masked_tk_id = self.get_masked_token()
            pseudo_ids.append(masked_tk_id)
            label_ids.append(tk_id.clone())
            r = random.random()
            if r < self.target_mask_prob:
                target_ids[pos] = 0
                target_ids[pos][masked_tk_id] = 1
        pseudo_ids = self.__pad(pseudo_ids, self.max_target_len)
        label_ids = torch.stack(label_ids, dim=0)


        p1d = (0, 0, 0, self.max_target_len - label_ids.shape[0])
        label_ids = F.pad(label_ids, p1d, "constant", 0)  # effectively zero padding

        p1d = (0, 0, 0, self.max_target_len - target_ids.shape[0])
        target_ids = F.pad(target_ids, p1d, "constant", 0)  # effectively zero padding

        return target_ids, label_ids, pseudo_ids

    def __getitem__(self, _idx) -> Seq2seqDatasetForBertElementType:
        if not self.soft_label:
            raise NotImplementedError("Not implemented yet.")
        
        return self.soft_label_getitem(_idx)

def batch_list_to_batch_tensors(batch: list[Seq2seqDatasetForBertElementType]):
    """
    (
        source_ids,
        target_ids,
        label_ids,
        pseudo_ids,
        num_source_tokens,
        num_target_tokens,
    )
    """
    return (
        torch.tensor([x[0] for x in batch], dtype=torch.long),
        torch.stack([x[1] for x in batch]),
        torch.stack([x[2] for x in batch]),
        torch.tensor([x[3] for x in batch], dtype=torch.long),
        torch.tensor([x[4] for x in batch], dtype=torch.long),
        torch.tensor([x[5] for x in batch], dtype=torch.long),
    )

def _fast_process_one(example, tokenizer):
    if isinstance(example["src"], list):
        source_tokens = example["src"]
        target_tokens = example["tgt"]
        if "tgt_kd" in example:
            target_kd_tokens = example["tgt_kd"]
    else:
        source_tokens = tokenizer.tokenize(example["src"])
        target_tokens = tokenizer.tokenize(example["tgt"])
        if "tgt_kd" in example:
            target_kd_tokens = tokenizer.tokenize(example["tgt_kd"])
    out_dict = {
            "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
            "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
        }
    if "tgt_kd" in example:
        out_dict["target_kd_ids"] = tokenizer.convert_tokens_to_ids(target_kd_tokens)
    return out_dict

def deserialize_str(x):
    return x.decode('ascii')

class DocDB(object):
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.start_key_index = int(deserialize_str(txn.get(b'__start__')))
            self.size = int(deserialize_str(txn.get(b'__size__')))
            self.dtype = deserialize_str(txn.get(b'__dtype__'))

    def _deserialize_array(self, x):
        data = array.array(self.dtype)
        data.frombytes(x)
        return data.tolist()

    def __getitem__(self, doc_id):
        with self.env.begin(write=False) as txn:
            example = TrainingExample(
                source_ids=self._deserialize_array(txn.get(b"src_ids_%d" % doc_id)), 
                target_ids=self._deserialize_array(txn.get(b"tgt_ids_%d" % doc_id)),
                example_id=None, 
            )
        return example

    def __len__(self):
        return self.size

def report_length(length_counter, total_count):
    max_len = max(length_counter.keys())
    a = 0
    tc = 0
    while a < max_len:
        cc = 0
        for i in range(16):
            cc += length_counter[a + i]

        tc += cc
        if cc > 0:
            logger.info("%d ~ %d = %d, %.2f%%" % (a, a + 16, cc, (tc * 100.0) / total_count))
        a += 16

def write_to_lmdb(db, key, value):
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key, value)
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            print('>>> Doubling LMDB map size to %sMB ...' %
                  (new_limit >> 20,))
            db.set_mapsize(new_limit)  # double it

def serialize_array(x, dtype):
    data = array.array(dtype)  # <- Deprecated, but I don't know how to fix this.
    data.fromlist(x)
    return data.tobytes()

def serialize_str(x):
    return u"{}".format(x).encode('ascii')

def load_and_cache_examples(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=True, 
        lmdb_cache=None, lmdb_dtype='h', eval_mode=False, soft_label=False):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.isfile(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features: list[TrainingExample] = torch.load(cached_features_file, weights_only=False)
    elif cached_features_file is not None and os.path.isdir(cached_features_file) \
        and os.path.exists(os.path.join(cached_features_file, 'lock.mdb')):
        raise NotImplementedError("Seriously, I have no idea for this.")
        logger.info("Loading features from cached LMDB %s", cached_features_file)
        features = DocDB(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        examples = []
        with open(example_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                examples.append(json.loads(line))
        features: list[TrainingExample] = []

        slc = defaultdict(int)
        tlc = defaultdict(int)

        for example in tqdm(examples):
            if soft_label:
                source_tokens = tokenizer.tokenize(example["src"])
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                target_ids = [tokenizer.convert_tokens_to_ids([
                    id_to_label_token(label).lower() for label in label_list_per_level
                ]) for label_list_per_level in example['tgt']]
            else:
                raise ValueError("For soft_label = False, I didn't support it for now.")

            slc[len(source_ids)] += 1
            tlc[len(target_ids)] += 1

            features.append(
                TrainingExample(
                    source_ids=source_ids,
                    target_ids=target_ids,
                    example_id=len(features), 
                )
            )

        if shuffle:
            random.shuffle(features)
            logger.info("Shuffle the features !")

        logger.info("Source length:")
        report_length(slc, total_count=len(examples))
        logger.info("Target length:")
        report_length(tlc, total_count=len(examples))

        if local_rank in [-1, 0] and cached_features_file is not None:
            if lmdb_cache:
                raise NotImplementedError("Not support it yet.")
            else:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features

def load_and_cache_examples_fast(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=True, eval_mode=False):
    """
    https://github.com/kongds/HBGL/blob/master/test.py
    """
    
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()


    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file, weights_only=False)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        examples = []
        with open(example_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                examples.append(json.loads(line))

        logger.info("Staring processing")
        b = time.time()
        with Pool() as p:
            features = p.starmap(_fast_process_one, [(i, tokenizer) for i in examples])

        features = [
            TrainingExample(
                source_ids=f['source_ids'],
                target_ids=f['target_ids'] if not eval_mode else [],
                example_id=i,
            ) for i, f in enumerate(features)
        ]

        b = time.time() - b
        logger.info("End took %s s", b)


        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features

def id_to_label_token(id_: int):
    return f"[A_{id_}]"
