import glob
import logging
import os
import shutil

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import wandb

import test
from utils import Seq2seqDatasetForBert, batch_list_to_batch_tensors

logger = logging.getLogger(__name__)

OPTIM_NAME = "optimizer.bin"
WEIGHTS_NAME = 'pytorch_model.bin'

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

def get_checkpoint_state_dict(output_dir, ckpt):
    model_recover_checkpoint = os.path.join(output_dir, "ckpt-%d" % ckpt, WEIGHTS_NAME)
    logger.info(" ** Recover model checkpoint in %s ** ", model_recover_checkpoint)
    model_state_dict = torch.load(model_recover_checkpoint, map_location='cpu')
    optimizer_recover_checkpoint = os.path.join(output_dir, "ckpt-%d" % ckpt, OPTIM_NAME)
    checkpoint_state_dict = torch.load(optimizer_recover_checkpoint, map_location='cpu')
    checkpoint_state_dict['model'] = model_state_dict
    return checkpoint_state_dict

def prepare_for_training(args, model, checkpoint_state_dict):
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
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    return model, optimizer

def train(
        args,
        training_features,
        model,
        tokenizer
):
    """
    https://github.com/kongds/HBGL/blob/master/run.py

    Train the model
    
    Removed some features:
    - tb_writer (involving SummaryWriter, idk what is it)
    - args.fp16 (amp)
    """
    tb_writer = None
    amp = None

    # model recover
    recover_step = get_max_epoch_model(args.output_dir)

    if recover_step:
        checkpoint_state_dict = get_checkpoint_state_dict(args.output_dir, recover_step)
    else:
        checkpoint_state_dict = None

    model.to(args.device)
    model, optimizer = prepare_for_training(args, model, checkpoint_state_dict)

    per_node_train_batch_size = args.per_gpu_train_batch_size * args.n_gpu * args.gradient_accumulation_steps
    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    global_step = recover_step if recover_step else 0

    if args.num_training_steps == -1:
        args.num_training_steps = args.num_training_epochs * len(training_features) / train_batch_size

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_training_steps, last_epoch=-1)

    if checkpoint_state_dict:
        scheduler.load_state_dict(checkpoint_state_dict["lr_scheduler"])

    train_dataset = create_train_dataset(args, training_features, model, tokenizer, train_batch_size, global_step)

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_features))
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.num_training_steps)

    if args.num_training_steps <= global_step:
        logger.info("Training is done. Please use a new dir or clean this dir!")
    else:
        # The training features are shuffled
        train_sampler = SequentialSampler(train_dataset) \
            if args.local_rank == -1 else DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size // args.gradient_accumulation_steps,
            collate_fn=batch_list_to_batch_tensors)

        train_iterator = tqdm(
            train_dataloader, initial=global_step * args.gradient_accumulation_steps,
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=args.local_rank not in [-1, 0])

        model.train()
        model.zero_grad()

        tr_loss, logging_loss = 0.0, 0.0
        best_macro_f1, best_micro_f1 = 0, 0
        best_macro_f1_path, best_micro_f1_path = None, None

        for step, batch in enumerate(train_iterator):
            if global_step > args.num_training_steps:
                break
            batch = tuple(t.to(args.device) for t in batch)
            if args.mask_way == 'v2':
                inputs = {'source_ids': batch[0],
                        'target_ids': batch[1],
                        'label_ids': batch[2],
                        'pseudo_ids': batch[3],
                        'num_source_tokens': batch[4],
                        'num_target_tokens': batch[5]}
            elif args.mask_way == 'v1' or args.mask_way == 'v0':
                inputs = {'source_ids': batch[0],
                        'target_ids': batch[1],
                        'masked_ids': batch[2],
                        'masked_pos': batch[3],
                        'masked_weight': batch[4],
                        'num_source_tokens': batch[5],
                        'num_target_tokens': batch[6]}

            if args.soft_label:
                inputs['label_ids'] = inputs['label_ids'].float()
                inputs['target_ids'] = inputs['target_ids'].float()

            if args.label_cpt_decodewithpos:
                model.target_offset = args.max_source_seq_length
                inputs['target_no_offset'] = True

            loss = model(**inputs)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_lr()[0]))
            if args.wandb:
                if (step + 1) % 50 == 0:
                    wandb.log({'train/loss': loss.item()})
                    wandb.log({'train/learning_rate': scheduler.get_lr()[0],
                                   "train/global_step": step})
            else:
                if (step + 1) % 50 == 0:
                    print('train/loss', loss.item())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()


            logging_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("")
                    logger.info(" Step [%d ~ %d]: %.2f", global_step - args.logging_steps, global_step, logging_loss)
                    logging_loss = 0.0

                if args.local_rank in [-1, 0] and args.save_steps > 0 and \
                        (global_step % args.save_steps == 0 or global_step == args.num_training_steps):

                    save_path = save_step(args, model, optimizer, global_step, scheduler)

                    flags = [
                    '--tokenizer_name'         , args.model_name_or_path             ,
                     '--input_file'             , args.valid_file                  ,
                     '--split'                  , 'valid'                         ,
                     '--do_lower_case'          ,
                     '--model_path'             , str(save_path)              ,
                     '--max_seq_length'         , str(args.max_source_seq_length + args.max_target_seq_length) if args.label_cpt_decodewithpos else str(args.max_source_seq_length)             ,
                     '--max_tgt_length'         , str(args.max_target_seq_length)             ,
                     '--batch_size'             , '32'                            ,
                     '--beam_size'              , '1'                             ,
                     '--length_penalty'         , '0'                             ,
                     '--forbid_duplicate_ngrams',
                     '--mode'                   , 's2s'                           ,
                     '--forbid_ignore_word'     , '"."'                           ,
                     '--cached_features_file'   , str(os.path.join(args.output_dir, "cached_features_for_valid.pt")),
                     '--add_vocab_file'         , args.add_vocab_file]

                    if args.softmax_label_only:
                        flags.append('--softmax_label_only')
                    if args.soft_label:
                        flags.append('--soft_label')
                    if args.soft_label_hier_real:
                        flags.append('--soft_label_hier_real_with_train_file')
                        flags.append(args.train_file)
                    if args.label_cpt_decodewithpos:
                        flags.append('--target_no_offset')

                    out = test.main(flags)
                    if out:
                        if args.wandb:
                            wandb.log({'eval/macro_f1': out['macro_f1'], 'eval/micro_f1': out['micro_f1']})

                        keep_save_model = False
                        if out['macro_f1'] > best_macro_f1:
                            best_macro_f1 = out['macro_f1']
                            if best_macro_f1_path != best_micro_f1_path and best_macro_f1_path is not None:
                                try:
                                    shutil.rmtree(best_macro_f1_path)
                                except:
                                    pass
                            best_macro_f1_path = save_path
                            keep_save_model = True

                        if out['micro_f1'] > best_micro_f1:
                            best_micro_f1 = out['micro_f1']
                            if best_micro_f1_path != best_macro_f1_path and best_micro_f1_path is not None:
                                try:
                                    shutil.rmtree(best_micro_f1_path)
                                except:
                                    pass
                            best_micro_f1_path = save_path
                            keep_save_model = True

                        if not keep_save_model:
                            try:
                                shutil.rmtree(save_path)
                            except:
                                pass
                    
                    print('best micro', best_micro_f1_path, best_micro_f1)
                    print('best macro', best_macro_f1_path, best_macro_f1)

    if args.local_rank in [-1, 0] and tb_writer:
        tb_writer.close()
    return best_macro_f1_path, best_micro_f1_path

def save_step(args, model, optimizer, global_step, scheduler):
    save_path = os.path.join(args.output_dir, "ckpt-%d" % global_step)
    os.makedirs(save_path, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_path)

    optim_to_save = {
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": scheduler.state_dict(),
                    }
    torch.save(optim_to_save, os.path.join(save_path, OPTIM_NAME))
    logger.info("Saving model checkpoint %d into %s", global_step, save_path)
    return save_path

def create_train_dataset(args, training_features, model, tokenizer, train_batch_size, global_step):
    train_dataset = Seq2seqDatasetForBert(
        features=training_features, max_source_len=args.max_source_seq_length,
        max_target_len=args.max_target_seq_length, vocab_size=model.bert.embeddings.word_embeddings.num_embeddings,
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id,
        mask_id=tokenizer.mask_token_id, random_prob=args.random_prob, keep_prob=args.keep_prob,
        offset=train_batch_size * global_step, num_training_instances=train_batch_size * args.num_training_steps,
        source_mask_prob=args.source_mask_prob, target_mask_prob=args.target_mask_prob,
        mask_way=args.mask_way, num_max_mask_token=args.num_max_mask_token,
        soft_label=args.soft_label,
    )

    logger.info("Check dataset:")
    for i in range(5):
        ids = train_dataset.__getitem__(i)
        if ids is None:
            raise ValueError(f"No data at index={i}")

        source_ids, target_ids = ids[:2]
        logger.info("Instance-%d" % i)
        logger.info("Source tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(source_ids)))
        if args.soft_label:
            real_target_ids = []
            if type(target_ids) is list:
                target_ids = torch.tensor(target_ids)
            for i in range(target_ids.shape[0]):
                real_target_ids.append(torch.arange(target_ids.shape[-1])[target_ids[i].bool()].tolist())
            for rti in real_target_ids:
                logger.info("Target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(rti)))
        else:
            logger.info("Target tokens = %s" % " ".join(tokenizer.convert_ids_to_tokens(target_ids)))

    logger.info("Mode = %s" % str(model))
    return train_dataset
