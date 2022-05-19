""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from model.modeling_albert import AlbertConfig
# from model.modeling_albert_bright import AlbertConfig, AlbertForSequenceClassification # chinese version
from model import tokenization_albert
from model.file_utils import WEIGHTS_NAME
from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from model.tokenization_bert import BertTokenizer
from metrics.glue_compute_metrics import compute_metrics
from processors import glue_output_modes as output_modes
from processors import glue_processors as processors
from processors import glue_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn
from tools.common import seed_everything
from tools.common import init_logger, logger
from callback.progressbar import ProgressBar
from model.modeling_albert import AlbertMyClassifier,AlbertPreClassifier,AlbertForSequenceClassification

def load_and_cache_examples(task, tokenizer,data_dir,model_name_or_path,max_seq_length,local_rank,data_type='train'):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, model_name_or_path.split('/'))).pop(),
        str(max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s",data_dir)
        label_list = processor.get_labels()

        if data_type == 'train':
            examples = processor.get_train_examples(data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(data_dir)
        else:
            examples = processor.get_test_examples(data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_seq_length=max_seq_length,
                                                output_mode = output_mode)
        if local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)


    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset

data_dir='dataset/tnews'
task_name='tnews'

model_name_or_path='pre_trained_model/albert_small_zh'

tokenizer=BertTokenizer.from_pretrained(model_name_or_path)

train_dataset=load_and_cache_examples(task_name,tokenizer,data_dir,model_name_or_path,100,-1)


per_gpu_train_batch_size=128
local_rank=-1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
max_steps=-1
gradient_accumulation_steps=1
warmup_proportion=0.1
weight_decay=0.0
learning_rate=5e-5
classifier_learning_rate=1e-3
adam_epsilon=1e-6
seed=42
max_grad_norm=1.0
save_steps=5000
logging_steps=5000
max_sequence_length=50

output_mode='classification'
# print(f'n_gpu:{n_gpu}')
# train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
# print(f'train_batch_size:{train_batch_size}')



"""train"""
def train(train_dataset, model, tokenizer,per_gpu_train_batch_size,local_rank,device,n_gpu,max_steps,gradient_accumulation_steps,
          num_train_epochs,warmup_proportion,weight_decay,learning_rate,classifier_learning_rate,adam_epsilon,seed,max_grad_norm,save_steps,
          logging_steps,model_name_or_path,data_dir,output_dir,max_sequence_length,output_mode):
    """ Train the model """
    train_batch_size =per_gpu_train_batch_size * max(1,n_gpu)
    train_sampler = RandomSampler(train_dataset) if local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,
                                  collate_fn=collate_fn)

    """
    max_steps:自定义的模型更新次数
    1.存在max_steps时，模型的总训练步数=自定义的模型更新次数
        训练的epoch数=(模型总训练步数//一个epoch内模型的训练次数)+1=max_steps//(batch数//梯度累计数)+1
    2.不存在max_steps时，根据epochs数计算模型的总训练步数
        总训练步数=一个epoch内模型更新的次数*epoch数=(batch数/梯度累计步数)*epoch数
    """
    if max_steps > 0:
        num_training_steps = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) //gradient_accumulation_steps) + 1
    else:
        num_training_steps = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    """确定warm_up数，即更新步数到哪一步时，学习率开始衰减=模型总的训练不是*warm_up的比例"""
    warmup_steps = int(num_training_steps * warmup_proportion)

    # Prepare optimizer and schedule (linear warmup and decay)
    """确定不参与权重衰减的参数，和参与权重衰减的参数"""
    no_decay = ['bias', 'LayerNorm.weight']
    classifier_params=['classifier']
    optimizer_grouped_parameters = [
        #1.classifier层的weight：参与衰减，学习率=classifier的学习率
        {'params': [p for n,p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay,'lr':classifier_learning_rate},
        #2.classifier层bias:不参与权重衰减，学习率：classifier的学习率
        {'params': [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': classifier_learning_rate},
        #3.其它层的bias和layernormweight：不参与权重衰减，学习率：默认学习率
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in classifier_params) and
                    any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': learning_rate},
        #4.其它层的weight：参与权重衰减，学习率：默认学习率
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in classifier_params) and
                    not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': learning_rate}
    ]
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                train_batch_size * gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.to(device)
    model.zero_grad()
    seed_everything(seed)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(int(num_train_epochs)):
        print(f'epoch{epoch}')
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss /gradient_accumulation_steps

            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            """到达梯度累计步数时，进行梯度更新、学习率更新，再将梯度清零"""
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            # """当前步数为记录步数的整数倍时，对当前模型进行评估"""
            # if local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0:
            #     # Log metrics
            #     if local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
            #         evaluate(model,'tnews',output_dir,data_dir,model_name_or_path,max_sequence_length,
            #                  local_rank,tokenizer,per_gpu_train_batch_size,n_gpu,device,output_mode)
            pbar(step, {'loss': loss.item()})

        """当前步数为存储步数的整数倍时，对当前模型进行存储"""
        if local_rank in [-1, 0] and save_steps > 0:
            # Save model checkpoint
            """存储当前更新步数下的模型"""
            output_dir_current = os.path.join(output_dir, 'checkpoint-{}'.format(epoch))
            if not os.path.exists(output_dir_current):
                os.makedirs(output_dir_current)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir_current)
            # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir_current)

        print(" ")
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


processor=processors[task_name]()
num_labels=len(processor.get_labels())
vocab_file='pre_trained_model/albert_small_zh/vocab.txt'
do_lower_case=True
config = AlbertConfig.from_pretrained(model_name_or_path,
                                      num_labels=num_labels,
                                      finetuning_task=task_name)
tokenizer = tokenization_albert.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
model =AlbertPreClassifier.from_pretrained(model_name_or_path,
                                                       from_tf=bool('.ckpt' in model_name_or_path),
                                                        config=config)





def evaluate(model, task_name,output_dir,data_dir,model_name_or_path,max_sequence_length,local_rank,
             tokenizer, per_gpu_eval_batch_size,n_gpu,device,output_mode,data_type='dev',prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_name =task_name
    eval_output_dir =output_dir

    results = {}
    eval_dataset = load_and_cache_examples(task_name,tokenizer,data_dir,model_name_or_path,
                                           max_sequence_length,local_rank,data_type=data_type)
    if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_batch_size =per_gpu_eval_batch_size * max(1,n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    model.to(device)
    model.eval()
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):

        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        pbar(step)
    print(' ')
    if 'cuda' in str(device):
        torch.cuda.empty_cache()
    eval_loss = eval_loss / nb_eval_steps
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(eval_task_name, preds, out_label_ids)
    print(result)
    results.update(result)
    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    return results


# model_name_or_path='outputs/tnews_output_premethod_onelayer/checkpoint-5000/checkpoint-10000/checkpoint-15000'
# model =AlbertPreClassifier.from_pretrained(model_name_or_path,
#                                                        from_tf=bool('.ckpt' in model_name_or_path),
#                                                         config=config)
# output_dir='outputs/tnews_output_premethod_twolayer'
# evaluate(model,'tnews',output_dir,data_dir,model_name_or_path,max_sequence_length,local_rank,tokenizer,
#          per_gpu_train_batch_size,n_gpu,device,output_mode)

is_train=True
is_evaluate=False
output_dir='outputs/tnews_output_premethod_two_layer'
num_train_epochs=30


if is_train==True:
    train(train_dataset,model,tokenizer,per_gpu_train_batch_size,local_rank,device,n_gpu, max_steps,
          gradient_accumulation_steps, num_train_epochs, warmup_proportion,
          weight_decay, learning_rate, classifier_learning_rate,adam_epsilon, seed, max_grad_norm,
          save_steps, logging_steps, model_name_or_path, data_dir, output_dir, max_sequence_length, output_mode)

if is_evaluate:
    results = []
    for root,dir,files in os.walk(output_dir):
        if files:
            print(root)
            current_epoch=root.split('\\')[-1]
            model=AlbertPreClassifier.from_pretrained(root)
            model.to(device)
            result =evaluate(model,task_name,output_dir,data_dir,model_name_or_path,max_sequence_length,local_rank,tokenizer,
                             per_gpu_train_batch_size,n_gpu,device,output_mode)
            results.extend([(k + '_{}'.format(current_epoch), v) for k, v in result.items()])
    output_eval_file=os.path.join(output_dir,"checkpoint_eval_results.txt")
    with open(output_eval_file, "w",encoding='utf-8') as writer:
        for key, value in results:
            if 'f1' not in key:
                writer.write("%s = %s\n" % (key, str(value)))



# if is_evaluate==True:
#     results = []
#     tokenizer = tokenization_albert.FullTokenizer(vocab_file=vocab_file,
#                                                   do_lower_case=do_lower_case)
#     checkpoints = [(0, output_dir)]
#
#     checkpoints = list(
#         os.path.dirname(c) for c in sorted(glob.glob(output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
#     checkpoints = [(int(checkpoint.split('-')[-1]), checkpoint) for checkpoint in checkpoints if
#                    checkpoint.find('checkpoint') != -1]
#     checkpoints = sorted(checkpoints, key=lambda x: x[0])
#     logger.info("Evaluate the following checkpoints: %s", checkpoints)
#     for _, checkpoint in checkpoints:
#         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
#         prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
#         model =AlbertPreClassifier.from_pretrained(checkpoint)
#         model.to(device)
#         result =evaluate(model,task_name,output_dir,data_dir,model_name_or_path,max_sequence_length,local_rank,tokenizer,
#                          per_gpu_train_batch_size,n_gpu,device,output_mode)
#         results.extend([(k + '_{}'.format(global_step), v) for k, v in result.items()])
#     output_eval_file = os.path.join(output_dir, "checkpoint_eval_results.txt")
#     with open(output_eval_file, "w") as writer:
#         for key, value in results:
#             writer.write("%s = %s\n" % (key, str(value)))



