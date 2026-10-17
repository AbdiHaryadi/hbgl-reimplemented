import glob
import logging
import math
import os
import shutil

import safetensors.torch
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import wandb

from args import HBGLTestingArguments, HBGLTrainingArguments
from modeling import BertForSequenceToSequence
import test_model
from utils import Seq2seqDatasetForBert, TrainingExample, batch_list_to_batch_tensors

logger = logging.getLogger(__name__)

OPTIM_NAME = "optimizer.bin"
WEIGHTS_NAME = 'model.safetensors'

def get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "ckpt-*/%s" % WEIGHTS_NAME))
    fn_optim_list = glob.glob(os.path.join(output_dir, "ckpt-*/%s" % OPTIM_NAME))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(os.path.dirname(fn).split('-')[-1]) for fn in fn_model_list]
                   ) & set([int(os.path.dirname(fn).split('-')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None

def load_model_state_dict(path):
    with open(path, "rb") as f:
        b = f.read()
    return safetensors.torch.load(b)

def get_checkpoint_state_dict(output_dir, ckpt):
    model_recover_checkpoint = os.path.join(output_dir, "ckpt-%d" % ckpt, WEIGHTS_NAME)
    logger.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint)
    model_state_dict = load_model_state_dict(model_recover_checkpoint)
    optimizer_recover_checkpoint = os.path.join(output_dir, "ckpt-%d" % ckpt, OPTIM_NAME)
    checkpoint_state_dict = torch.load(optimizer_recover_checkpoint, map_location='cpu')
    checkpoint_state_dict['model'] = model_state_dict
    return checkpoint_state_dict

def prepare_for_training(
        args: HBGLTrainingArguments,
        model: BertForSequenceToSequence,
        checkpoint_state_dict
):
    """
    https://github.com/kongds/HBGL/blob/master/run.py

    Remove:
    - amp things
    """

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if checkpoint_state_dict:
        optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        model.load_state_dict(checkpoint_state_dict['model'])

    # multi-gpu training
    if args.n_gpu > 1:
        new_model = torch.nn.DataParallel(model)
    else:
        new_model = model


    # Distributed training
    if args.local_rank != -1:
        new_model = torch.nn.parallel.DistributedDataParallel(
            new_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # else: no change

    return new_model, optimizer

class HBGLLossLogger:
    def __init__(self, args: HBGLTrainingArguments):
        self.args = args
        self.logging_loss = 0.0

    def accumulate(self, loss: float):
        self.logging_loss += loss

    def log_if_needed(self, global_step: int):
        if self.args.local_rank in [-1, 0] and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
            logger.info("")
            logger.info(" Step [%d ~ %d]: %.2f", global_step - self.args.logging_steps, global_step, self.logging_loss)
            self.logging_loss = 0.0

def delete_model(path: str):
    try:
        shutil.rmtree(path)
    except:
        print(f"Warning: Trying to remove path {path}, but can't")

class BestModelManager:
    def __init__(self) -> None:
        self.best_macro_f1_score = 0.0
        self.best_micro_f1_score = 0.0
        self.best_macro_f1_path: str | None = None
        self.best_micro_f1_path: str | None = None

    def handle(self, new_path: str, new_macro_f1_score: float | None, new_micro_f1_score: float | None):
        keep_save_model = False
        if new_macro_f1_score is not None and new_macro_f1_score > self.best_macro_f1_score:
            self.best_macro_f1_score = new_macro_f1_score
            if self.best_macro_f1_path != self.best_micro_f1_path and self.best_macro_f1_path is not None:
                delete_model(self.best_macro_f1_path)
            self.best_macro_f1_path = new_path
            keep_save_model = True

        if new_micro_f1_score is not None and new_micro_f1_score > self.best_micro_f1_score:
            self.best_micro_f1_score = new_micro_f1_score
            if self.best_macro_f1_path != self.best_micro_f1_path and self.best_micro_f1_path is not None:
                delete_model(self.best_micro_f1_path)
            self.best_micro_f1_path = new_path
            keep_save_model = True

        return keep_save_model

class GlobalStepManager:
    def __init__(self, args: HBGLTrainingArguments):
        recover_step = get_max_epoch_model(args.output_dir)
        if recover_step is None:
            self.global_step = 0
        else:
            self.global_step = recover_step

    def get(self):
        return self.global_step

    def increment(self):
        self.global_step += 1

class HBGLTrainer:
    def __init__(self, args: HBGLTrainingArguments):
        self.args = args
        self.loss_logger = HBGLLossLogger(args)
        self.model_manager = BestModelManager()
        self.global_step_manager = GlobalStepManager(args)

    def get_current_global_step(self):
        return self.global_step_manager.get()

    def increment_global_step(self):
        self.global_step_manager.increment()

    def train(
            self,
            training_features: list[TrainingExample],
            base_model: BertForSequenceToSequence,
            tokenizer: BertTokenizer,
    ):
        """
        https://github.com/kongds/HBGL/blob/master/run.py
    
        Train the model
        
        Removed some features:
        - tb_writer (involving SummaryWriter, idk what is it)
        - args.fp16 (amp)
        """
        # model recover
        recover_step = get_max_epoch_model(self.args.output_dir)
    
        if recover_step:
            checkpoint_state_dict = get_checkpoint_state_dict(
                self.args.output_dir,
                recover_step
            )
        else:
            checkpoint_state_dict = None
    
        base_model.to(self.args.device)  # type: ignore
        model, optimizer = prepare_for_training(self.args, base_model, checkpoint_state_dict)
    
        per_node_train_batch_size = self.args.per_gpu_train_batch_size * self.args.n_gpu * self.args.gradient_accumulation_steps
        train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
    
        if self.args.num_training_steps == -1:
            self.args.num_training_steps = self.args.num_training_epochs * math.ceil(len(training_features) / train_batch_size)
    
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.num_training_steps, last_epoch=-1)
    
        if checkpoint_state_dict:
            scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])
    
        train_dataset = create_train_dataset(
            self.args,
            training_features,
            to_base_model(model),
            tokenizer,
            train_batch_size,
            self.get_current_global_step(),
        )
    
        # Train!
        logger.info("  ***** Running training *****  *")
        logger.info("  Num examples = %d", len(training_features))
        logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_features))
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Batch size per node = %d", per_node_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.args.num_training_steps)
    
        if self.args.num_training_steps <= self.get_current_global_step():
            logger.info("Training is done. Please use a new dir or clean this dir!")
        else:
            # The training features are shuffled
            train_sampler = SequentialSampler(train_dataset) \
                if self.args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
            train_dataloader = DataLoader(
                train_dataset, sampler=train_sampler,
                batch_size=per_node_train_batch_size // self.args.gradient_accumulation_steps,
                collate_fn=batch_list_to_batch_tensors)
    
            train_iterator = tqdm(
                train_dataloader, initial=self.get_current_global_step() * self.args.gradient_accumulation_steps,
                desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=self.args.local_rank not in [-1, 0])
    
            model.train()
            model.zero_grad()
    
            for step, batch in enumerate(train_iterator):
                if self.get_current_global_step() > self.args.num_training_steps:
                    break
                batch = tuple(t.to(self.args.device) for t in batch)
                (
                    source_ids,
                    target_ids,
                    label_ids,
                    pseudo_ids,
                    num_source_tokens,
                    num_target_tokens
                ) = tuple(t.to(self.args.device) for t in batch)
    
                # mask_way == 'v2'
                loss = model(
                    source_ids=source_ids,
                    target_ids=target_ids.float(),
                    label_ids=label_ids.float(),
                    pseudo_ids=pseudo_ids,
                    num_source_tokens=num_source_tokens,
                    num_target_tokens=num_target_tokens,
                )
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
    
                train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
                if self.args.wandb:
                    if (step + 1) % 50 == 0:
                        wandb.log({'train/loss': loss.item()})
                        wandb.log({'train/learning_rate': scheduler.get_lr()[0],
                                        "train/global_step": step})
                else:
                    if (step + 1) % 50 == 0:
                        print('train/loss', loss.item())
    
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
    
                loss.backward()
                self.loss_logger.accumulate(loss.item())

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
    
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    self.increment_global_step()
    
                    self.log_loss_if_needed()
    
                    if self.args.local_rank in [-1, 0] and self.is_save_step_time():
                        save_path = self.save_step(to_base_model(model), optimizer, scheduler)
                        new_macro_f1, new_micro_f1  = self.evaluate_step(save_path)
                        if self.args.wandb:
                            wandb.log({'eval/macro_f1': new_macro_f1, 'eval/micro_f1': new_micro_f1})

                        keep_save_model = self.model_manager.handle(
                            save_path,
                            new_macro_f1_score=new_macro_f1,
                            new_micro_f1_score=new_micro_f1
                        )
                        if not keep_save_model:
                            delete_model(save_path)
                                    
                        self.after_evaluate_step()
    
        return self.model_manager.best_macro_f1_path, self.model_manager.best_micro_f1_path

    def is_save_step_time(self):
        return self.args.save_steps > 0 and (
            self.get_current_global_step() % self.args.save_steps == 0
            or self.get_current_global_step() == self.args.num_training_steps
        )

    def save_step(
            self,
            model: BertForSequenceToSequence,
            optimizer: AdamW,
            scheduler: torch.optim.lr_scheduler.LRScheduler
    ):
        save_path = os.path.join(self.args.output_dir, "ckpt-%d" % self.get_current_global_step())
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
    
        optim_to_save = {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
        }
        torch.save(optim_to_save, os.path.join(save_path, OPTIM_NAME))
        logger.info("Saving model checkpoint %d into %s", self.get_current_global_step(), save_path)
        return save_path

    def log_loss_if_needed(self):
        self.loss_logger.log_if_needed(self.get_current_global_step())

    def after_evaluate_step(self):
        print('best micro', self.model_manager.best_micro_f1_path, self.model_manager.best_micro_f1_score)
        print('best macro', self.model_manager.best_macro_f1_path, self.model_manager.best_macro_f1_score)

    def evaluate_step(self, save_path: str):
        if self.args.add_vocab_file is None:
            return None, None
    
        out = test_model.test(HBGLTestingArguments(
            tokenizer_name=self.args.model_name_or_path,
            input_file=self.args.valid_file,
            split="valid",
            model_path=save_path,
            max_seq_length=self.args.max_source_seq_length,
            max_tgt_length=self.args.max_target_seq_length,
            # Original: 32
            # Changing it to per_gpu_train_batch_size still CUDAOutOfMemoryError
            # Let's test it only with one.
            batch_size=self.args.per_gpu_train_batch_size,  
            cached_features_file=os.path.join(
                self.args.output_dir,
                "cached_features_for_valid.pt"
            ),
            add_vocab_file=self.args.add_vocab_file,
            soft_label_hier_real_with_train_file=(
                self.args.train_file if self.args.soft_label_hier_real
                else None
            ),
        ))

        if out:
            return out['macro_f1'], out['micro_f1']
        else:
            return None, None

class InterruptableHBGLTrainer(HBGLTrainer):
    def __init__(self, args: HBGLTrainingArguments, interrupt_steps: int):
        super().__init__(args)
        self.interrupt_steps = interrupt_steps

    def after_evaluate_step(self):
        super().after_evaluate_step()
        if self.get_current_global_step() >= self.interrupt_steps:
            raise ValueError("Interrupt!")

def to_base_model(model: BertForSequenceToSequence | torch.nn.DataParallel[BertForSequenceToSequence] | torch.nn.parallel.DistributedDataParallel):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        base_model = model.module
        assert isinstance(base_model, BertForSequenceToSequence)
        return base_model
    
    return model

def create_train_dataset(
        args: HBGLTrainingArguments,
        training_features: list[TrainingExample],
        model: BertForSequenceToSequence,
        tokenizer: BertTokenizer,
        train_batch_size: int,
        global_step: int
):
    train_dataset = Seq2seqDatasetForBert(
        features=training_features,
        max_source_len=args.max_source_seq_length,
        max_target_len=args.max_target_seq_length,
        vocab_size=model.bert.embeddings.word_embeddings.num_embeddings,
        tokenizer=tokenizer,
        offset=train_batch_size * global_step,
        num_training_instances=train_batch_size * args.num_training_steps,
        source_mask_prob=args.source_mask_prob,
        target_mask_prob=args.target_mask_prob,
    )

    logger.info("Check dataset:")
    for i in range(5):
        source_ids, target_ids, _, _, _, _ = train_dataset[i]
        logger.info("Instance-%d" % i)
        logger.info("Source tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(source_ids)))
        real_target_ids = []
        for i in range(target_ids.shape[0]):
            real_target_ids.append(torch.arange(target_ids.shape[-1])[target_ids[i].bool()].tolist())
        for rti in real_target_ids:
            logger.info("Target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(rti)))

    logger.info("Mode = %s" % str(model))
    return train_dataset
