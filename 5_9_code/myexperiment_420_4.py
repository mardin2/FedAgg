from __future__ import absolute_import, division, print_function

import argparse
import os
import copy
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model.modeling_albert import AlbertConfig
from callback.optimization.adamw import AdamW
from processors import glue_processors as processors
from processors import collate_fn
from tools.common import seed_everything
from tools.common import init_logger, logger
from utils import compute_metrics
from processors.dataset_partition import load_and_cache_examples,load_dataset,partition
from utils import init_logger
from callback.progressbar import ProgressBar
from model.tokenization_albert import FullTokenizer
from model.modeling_albert import AlbertMyClassifier

# task='tnews'
# model_name_or_path='../outputs/tnews_output/checkpoint-500'
# vocab_file='../prev_trained_model/albert_small_zh/vocab.txt'
# do_lower_case=True
# processor=processors['tnews']()
# data_dir='../dataset/tnews'
# max_sequence_length=50
# local_rank=-1
# num_labels=len(processor.get_labels())
# fed_learning_rate,fed_adam_epsilon,fed_num_train_epochs,device,fed_max_grad_norm=0.01,1e-6,3,torch.device("cuda" if torch.cuda.is_available() else "cpu"),1.0
# eval_batch_size=32
# data_dir='../dataset/tnews'
# task='tnews'
# num_parties=4
# max_seq_length=50


# config = AlbertConfig.from_pretrained(model_name_or_path,
#                                       num_labels=num_labels,
#                                       finetuning_task=task)
# tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
# model = AlbertMyClassifier.from_pretrained(model_name_or_path,
#                                            from_tf=bool('.ckpt' in model_name_or_path),
#                                            config=config)
# party_dataset=iid_quantity(num_parties,data_dir,task,tokenizer,max_seq_length,local_rank)
#
# global_dataset=load_and_cache_examples(task,tokenizer,data_dir,model_name_or_path,max_seq_length,
#                                        local_rank,data_type='test')

def init_models(args,n_parties,model_init):
    """
    ???????????????????????????????????????????????????????????????????????????
    """
    #model_name_or_path:albert_pytorch_master/outputs/tnews_output
    models={model_i:None for model_i in range(n_parties)}
    for model_i in range(args.num_parties):
        # if args.task_name=='tnews':
        models[model_i]=copy.deepcopy(model_init)
    return models


"""1.????????????party????????????????????????"""
"""?????????fedavg??????????????????party??????????????????????????????"""
def train_model(model_id,train_dataloader, model,args):
    """ Train the model """
    logger.info('Training network %s'%str(model_id))
    classifier_params=['classifier']
    optimizer_group_parameters=[
        #1.albert???????????????????????????
        {'params': [p for n,p in model.classifier.named_parameters()],'lr':args.classifier_lr},
        #2.????????????????????????
        {'params':[p for n, p in model.named_parameters() if not any(nd in n for nd in classifier_params)],
         'lr':args.fed_lr}
    ]
    optimizer = AdamW(params=optimizer_group_parameters,eps=args.fed_adam_epsilon)
    # optimizer=torch.optim.SGD(params=optimizer_group_parameters,momentum=0.9)
    global_step = 0
    tr_loss= 0.0
    model.to(args.device)
    # model.zero_grad()
    for epoch in range(int(args.fed_num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        #????????????epoch???loss
        epoch_loss_collector=[]
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss= outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(),args.fed_max_grad_norm)
            tr_loss += loss.item()
            optimizer.step()
            # model.zero_grad()
            optimizer.zero_grad()
            global_step+=1
            epoch_loss_collector.append(loss.item())
            pbar(step, {'loss': loss.item()})
        print(" ")
        #??????epoch???????????????
        epoch_loss=sum(epoch_loss_collector)/len(epoch_loss_collector)
        logger.info(f'Epoch:{epoch} Loss:{epoch_loss}')
    # print(f'????????????????????????')
    # print(f'model:{model_id}')
    # print(model.state_dict()['classifier.0.weight'])

    train_acc,train_f1=compute_metrics(model,train_dataloader,args.device)

    logger.info(f'Training accuracy:{train_acc}')
    logger.info(f'Training f1 score:{train_f1}')

    logger.info(f'Global Train Loss:{tr_loss/global_step}')

    logger.info('***Traning complete***')
    return train_acc,train_f1,(tr_loss/global_step)

"""??????train_model()??????????????????"""
# train_dataset_0=party_dataset[0]
# train_dataset_0=load_dataset(train_dataset_0)
# sampler=RandomSampler(train_dataset_0)
# train_dataloader_0=DataLoader(train_dataset_0,batch_size=32,sampler=RandomSampler(train_dataset_0),
#                               collate_fn=collate_fn)


"""2.????????????party????????????????????????"""
"""?????????fedavg??????????????????party??????????????????????????????"""
def train_local_model(args,models,selected,party_dataset):
    avg_train_acc=0.0
    non_zero_party = 0
    #global?????????dataloader
    # test_sampler = SequentialSampler(global_dataset)
    # test_dataloader_global = DataLoader(global_dataset,eval_batch_size,
    #sampler=test_sampler, collate_fn=collate_fn)
    for model_id,model in models.items():
        if model_id not in selected:
            continue
        ##########################################
        # print('??????????????????????????????')
        # print(models[17].state_dict()['classifier.0.weight'])
        # print(f'???????????????????????????:{model_id}')
        # print(models[model_id].state_dict()['classifier.0.weight'])
        ##########################################
        #?????????party??????????????????

        if len(party_dataset[model_id])>0:
            non_zero_party+=1
            local_dataset=party_dataset[model_id]
            logger.info(f'Training model:{model_id},n_training:{len(local_dataset)}')
            model.to(args.device)

            #party???train_dataloader
            local_dataset=load_dataset(local_dataset)
            train_sampler=RandomSampler(local_dataset)
            train_dataloader_local=DataLoader(local_dataset,args.fed_train_batch_size,
                                              sampler=train_sampler,collate_fn=collate_fn)

            train_acc,train_f1,train_avg_loss=train_model(model_id,train_dataloader_local,model,args)
            avg_train_acc+=train_acc

        ############################################
        # print('??????????????????????????????')
        # print(models[17].state_dict()['classifier.0.weight'])
        # print(f'???????????????????????????:{model_id}')
        # print(models[model_id].state_dict()['classifier.0.weight'])
        ###########################################
    if non_zero_party!=0:
        avg_train_acc/=non_zero_party

        if args.algorithm=='local_training':
            logger.info(f'avg train acc:{avg_train_acc}')
    else:
        avg_train_acc=0

    models_list=list(models.values())
    return models_list,avg_train_acc


"""??????train_local_model????????????"""
# alg='local_training'
# fed_train_batch_size=8
# sample=0.5
# arr=np.arange(num_parties)
# np.random.shuffle(arr)
# selected=arr[:int(num_parties*sample)]
# print(selected)


# models=init_models(num_parties,task,model)
# for model_id,model in models.items():
#     print(f'model_id:{model_id}')
#     print(f'model:{model}')
#     break
# print(f'models_values:{models.values()}')
# model_list=train_local_model(models,selected,party_dataset,global_dataset,alg,eval_batch_size,device,fed_train_batch_size,
#                              fed_learning_rate, fed_adam_epsilon, fed_num_train_epochs, fed_max_grad_norm)
# print(f'len_model_lisy:{len(model_list)}')
# print(f'model_list:{model_list}')


"""3.??????fedprox?????????????????????"""
def train_fedprox_model(model_id,global_model,train_dataloader, model,args):
    """ Train the model """
    logger.info('Training network %s'%str(model_id))

    server_model=copy.deepcopy(global_model).to(args.device)

    ##########################################
    # print(f'????????????global_param:')
    # print(global_model.state_dict()['classifier.0.weight'])
    # print(f'????????????server_param:')
    # print(server_model.state_dict()['classifier.0.weight'])
    ##########################################
    
    # for p in global_model.parameters():
    #   if p.requires_grad==True:
    #     p.requires_grad=False
    # for p in model.parameters():
    #   p.requires_grad=True
    # global_model=copy.deepcopy(global_model)
    classifier_params = ['classifier']
    optimizer_group_parameters = [
        # 1.albert???????????????????????????
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': args.classifier_lr},
        # 2.????????????????????????
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in classifier_params)],
         'lr': args.fed_lr}
    ]
    optimizer = AdamW(params=optimizer_group_parameters,eps=args.fed_adam_epsilon)
    # optimizer=torch.optim.SGD(params=optimizer_group_parameters,momentum=0.9)
    global_step = 0
    tr_loss= 0.0

    #???????????????????????????????????????????????????
    # global_weight_collector=list(global_model.to(args.device).parameters())
    # global_params=global_model.state_dict()
    model.to(args.device)
    model.train()
    # model.zero_grad()
    print(f'Training network party:{model_id}')
    for epoch in range(int(args.fed_num_train_epochs)):
        #????????????epoch???loss
        epoch_loss_collector=[]
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss= outputs[0]  # model outputs are always tuple in transformers (see doc)

            #fedprox?????????????????????????????????????????????????????????
            # for param_idx,param in enumerate(model.parameters()):
            #     fedprox_reg+=((args.mu/2)*torch.norm(param-global_weight_collector
            #                                    [param_idx])**2)
            fedprox_reg=0.0
            # for param_idx,param in enumerate(model.parameters()):
            #     fedprox_reg+=((args.mu/2)*torch.norm(param)**2)
            # for w,wt in zip(model.parameters(),global_param):
            #     fedprox_reg+=(w-wt).norm(2)


            # model_param=model.state_dict()
            # server_param=server_model.state_dict()
            # for key in model_param:
            #     fedprox_reg+=(args.mu/2)*(model_param[key]-server_param[key]).norm(2)
            for w, w_t in zip(model.parameters(), server_model.parameters()):
                fedprox_reg +=(args.mu/2)*(w - w_t).norm(2)
            # print(f'fedprox_reg:{fedprox_reg}')
            # print('server_param')
            # print(server_model.state_dict()['classifier.0.weight'])
            # print(f'fedprox_reg:{fedprox_reg}')

            # if step<3:
            #   print('model_params:')
            #   print(model.state_dict()['classifier.0.weight'])
            #   print('*'*20)
            #   print('global_params:')
            #   print(global_param['classifier.0.weight'])
            #   print('*'*20)
            #   print('global-model.norm(2)')
            #   print((global_model.state_dict()['classifier.0.weight']-model.state_dict()['classifier.0.weight']).norm(2))
            # print(f'fedprox_reg:{fedprox_reg}')
            loss=loss+fedprox_reg
            # print(f'loss:{loss}')
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(),args.fed_max_grad_norm)
            tr_loss += loss.item()
            optimizer.step()
            # model.zero_grad()
            optimizer.zero_grad()
            global_step+=1
            epoch_loss_collector.append(loss.item())
            pbar(step, {'loss': loss.item()})
        print(" ")
        #??????epoch???????????????
        epoch_loss=sum(epoch_loss_collector)/len(epoch_loss_collector)
        logger.info(f'Epoch:{epoch} Loss:{epoch_loss}')
    # print('********************')
    # print(f'model:{model_id}')
    # print(model.state_dict()['classifier.0.weight'])

    train_acc,train_f1=compute_metrics(model,train_dataloader,args.device)

    logger.info(f'Training accuracy:{train_acc}')
    logger.info(f'Training f1 score:{train_f1}')

    logger.info(f'Global Train Loss:{tr_loss/global_step}')

    logger.info('***Traning complete***')
    return train_acc,train_f1,(tr_loss/global_step)


"""????????????party????????????fedprox?????????????????????"""
def train_local_fedprox_model(models,global_model,selected,party_dataset,args):
    avg_train_acc=0.0
    non_zero_party=0
    #global?????????dataloader
    # test_sampler = SequentialSampler(global_dataset)
    # test_dataloader_global = DataLoader(global_dataset,eval_batch_size,
    #                                     sampler=test_sampler, collate_fn=collate_fn)
    for model_id,model in models.items():
        if model_id not in selected:
            continue
        # print('??????????????????????????????')
        # print(models[17].state_dict()['classifier.0.weight'])
        # print('??????????????????????????????')
        # print(models[model_id].state_dict()['classifier.0.weight'])
        #?????????party??????????????????

        if len(party_dataset[model_id])>0:
            non_zero_party+=1
            local_dataset=party_dataset[model_id]
            logger.info(f'Training model:{model_id},n_training:{len(local_dataset)}')
            model.to(args.device)

            #party???train_dataloader
            local_dataset=load_dataset(local_dataset)
            train_sampler=RandomSampler(local_dataset)
            train_dataloader_local=DataLoader(local_dataset,args.fed_train_batch_size,
                                          sampler=train_sampler,collate_fn=collate_fn)

            train_acc,train_f1,train_avg_loss=train_fedprox_model(model_id,global_model,train_dataloader_local,model,args)
            avg_train_acc+=train_acc
        # models[model_id]=model
        # print('?????????????????????????????????')
        # print(models[17].state_dict()['classifier.0.weight'])
        # print('???????????????????????????')
        # print(models[model_id].state_dict()['classifier.0.weight'])
    if non_zero_party!=0:
        avg_train_acc/=non_zero_party
    else:
        avg_train_acc=0

    if args.algorithm=='local_training':
        logger.info(f'avg train acc:{avg_train_acc}')

    models_list=list(models.values())
    return models_list,avg_train_acc



# seed=42
# alg='fedavg'
# is_same_initial=True
# communicate_round=4
# partition_strategy='iid_quantity'


def alg_fedavg(args,tokenizer,model_init):
    avg_train_acc_dict={}
    global_test_acc_dict={}
    test_acc=0
    seed_everything(args.seed)
    party_dataset = partition[args.partition_strategy](args,tokenizer)
    global_dataset = load_and_cache_examples(args,tokenizer,data_type='test')
    test_sampler = SequentialSampler(global_dataset)
    test_dataloader_global = DataLoader(global_dataset,args.fed_eval_batch_size,
                                        sampler=test_sampler, collate_fn=collate_fn)

    if args.algorithm == 'fedavg':
        logger.info('Initializeing models')
        models=init_models(args,args.num_parties,model_init)
        global_models=init_models(args,1,model_init)
        global_model=global_models[0]

        global_param=global_model.state_dict()
        if args.is_same_initial:
            for model_id,model in models.items():
                model.load_state_dict(global_param)

        for round in range(args.communication_rounds):
            logger.info(f'In Communication Round:{round}')

            arr=np.arange(args.num_parties)
            np.random.shuffle(arr)
            selected=arr[:int(args.num_parties*args.sample)]
            selected=sorted(selected)

            global_param=global_model.state_dict()
            ########################################
            # print(f'??????????????????????????????')
            # print(global_param['classifier.0.weight'])
            ########################################

            if round==0:
                if args.is_same_initial:
                    for idx in selected:
                        models[idx].load_state_dict(global_param)

            else:
                for idx in selected:
                    models[idx].load_state_dict(global_param)
            # print(f'global_param???:')
            # print(global_param['classifier.0.weight'])

            #????????????????????????

            models_list,avg_train_acc=train_local_model(args,models,selected,party_dataset)

            # print(f'?????????????????????????????????global_param:')
            # print(global_param['classifier.0.weight'])
            #?????????????????????????????????
            avg_train_acc_dict[round]=avg_train_acc

            #??????????????????
            #1.??????????????????????????????????????????
            total_num_data=sum(len(party_dataset[i]) for i in selected)
            if total_num_data!=0:
                #2.?????????????????????????????????party?????????????????????????????????/???????????????
                fed_avg_weight=[len(party_dataset[i])/total_num_data for i in selected]
                #3.???????????????????????????
                for idx in range(len(selected)):
                    model_param=models[selected[idx]].state_dict()
                    # print(f'idx:{selected[idx]}')
                    # print(model_param['classifier.0.weight'])
                    if idx==0:
                        for key in model_param:
                            global_param[key]=model_param[key]*fed_avg_weight[idx]
                        ########################################
                        # print(f'??????????????????{idx}???--model_param:')
                        # print(model_param['classifier.0.weight'])
                        ########################################
                    else:
                        for key in model_param:
                            global_param[key]+=model_param[key]*fed_avg_weight[idx]
                        ########################################
                        # print(f'??????????????????{idx}???--model_param:')
                        # print(model_param['classifier.0.weight'])
                        #########################################

            global_model.load_state_dict(global_param)
            ###########################################
            # print(f'??????????????????????????????')
            # print(global_param['classifier.0.weight'])
            ########################################
            if args.save_rounds>0 and (round+1)%args.save_rounds==0:
                #save global model checkpoint
                """??????????????????"""
                fed_output_dir=os.path.join(args.fed_output_dir,'checkpoint--{}--{}'.format(args.algorithm,round))
                if not os.path.exists(fed_output_dir):
                    os.makedirs(fed_output_dir)
                model_to_save=global_model.module if hasattr(global_model,'module') else global_model
                model_to_save.save_pretrained(fed_output_dir)
                logger.info('Saving global model checkpoint to %s',fed_output_dir)


            logger.info(f'global num_test:{len(global_dataset)}')
            test_acc,test_f1=compute_metrics(global_model,test_dataloader_global,args.device)
            global_test_acc_dict[round]=test_acc

            logger.info(f'global model test acc:{test_acc}')
            logger.info(f'global model test f1:{test_f1}')


        #?????????????????????????????????
        avg_train_acc_file=os.path.join(args.fed_output_dir,'{}-{}-train_acc.txt'.format(
            args.algorithm,args.partition_strategy))
        with open(avg_train_acc_file,'w') as f:
            f.writelines('algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                args.algorithm,args.partition_strategy,args.num_parties,args.sample,args.fed_num_train_epochs,args.communication_rounds))
            f.writelines('\n')
            for key,value in avg_train_acc_dict.items():
                f.writelines('round:{}--avg_train_acc:{}'.format(key,value))
                f.writelines('\n')

        #?????????????????????????????????
        global_test_acc_file=os.path.join(args.fed_output_dir,'{}-{}-test_acc.txt'.format(
            args.algorithm,args.partition_strategy))
        with open(global_test_acc_file,'w') as f:
            f.writelines(
                'algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                    args.algorithm, args.partition_strategy, args.num_parties, args.sample, args.fed_num_train_epochs,
                    args.communication_rounds))
            f.writelines('\n')
            for key,value in global_test_acc_dict.items():
                f.writelines('round:{}--global_test_acc:{}'.format(key,value))
                f.writelines('\n')

            
    return test_acc

# test_acc=alg_fedavg(seed,alg,partition_strategy,num_parties, data_dir, task, tokenizer, max_seq_length, local_rank,
#                 model_name_or_path, model, is_same_initial, communicate_round, sample, eval_batch_size,
#                 device, fed_train_batch_size, fed_learning_rate, fed_adam_epsilon, fed_num_train_epochs,
#                 fed_max_grad_norm)
#
# print(test_acc)


def alg_fedprox(args,tokenizer,model_init):

    avg_train_acc_dict={}
    global_test_acc_dict={} 

    test_acc=0
    seed_everything(args.seed)
    party_dataset = partition[args.partition_strategy](args,tokenizer)
    global_dataset = load_and_cache_examples(args,tokenizer,data_type='test')
    test_sampler = SequentialSampler(global_dataset)
    test_dataloader_global = DataLoader(global_dataset,args.fed_eval_batch_size,
                                        sampler=test_sampler, collate_fn=collate_fn)

    if args.algorithm == 'fedprox':
        logger.info('Initializeing models')
        models=init_models(args,args.num_parties,model_init)
        global_models=init_models(args,1,model_init)
        global_model=global_models[0]


        global_param=global_model.state_dict()
        if args.is_same_initial:
            for model_id,model in models.items():
                model.load_state_dict(global_param)

        for round in range(args.communication_rounds):
            logger.info(f'In Communication Round:{round}')

            arr=np.arange(args.num_parties)
            np.random.shuffle(arr)
            selected=arr[:int(args.num_parties*args.sample)]
            selected = sorted(selected)

            global_param=global_model.state_dict()
            # global_param_copy=copy.deepcopy(global_model.to(args.device).state_dict())

            ##################################
            # print(f'???????????????????????????global_param???')
            # print(global_param['classifier.0.weight'])
            # # print(f'??????copy?????????????????????global_param_copy:')
            # # print(global_param_copy['classifier.0.weight'])
            ##################################


            if round==0:
                if args.is_same_initial:
                    for idx in selected:
                        models[idx].load_state_dict(global_param)

            else:
                for idx in selected:
                    models[idx].load_state_dict(global_param)
            # print(f'global_param?????????')
            # print(global_param['classifier.0.weight'])
            

            models_list,avg_train_acc=train_local_fedprox_model(models,global_model,selected,party_dataset,args)
            ###################################
            # print(f'?????????????????????????????????global_param???')
            # print(global_param['classifier.0.weight'])
            # # print(f'??????????????????????????????global_param_copy:')
            # # print(global_param_copy['classifier.0.weight'])
            ###################################


            #?????????????????????????????????
            avg_train_acc_dict[round]=avg_train_acc


            #??????????????????
            #1.??????????????????????????????????????????
            total_num_data=sum(len(party_dataset[i]) for i in selected)
            if total_num_data!=0:
                #2.?????????????????????????????????party?????????????????????????????????/???????????????
                fed_avg_weight=[len(party_dataset[i])/total_num_data for i in selected]
                #3.???????????????????????????
                for idx in range(len(selected)):
                    # print('selected[idx]:{selected[idx]}')
                    model_param=models[selected[idx]].state_dict()
                    # print(model_param['classifier.0.weight'])
                    if idx==0:
                        for key in model_param:
                            global_param[key]=model_param[key]*fed_avg_weight[idx]
                        ####################################
                        # print(f'??????????????????{idx}???-model_param????????????????????????global_param???')
                        # print(f'weight:{fed_avg_weight[idx]}')
                        # print(model_param['classifier.0.weight'])
                        # print(global_param['classifier.0.weight'])
                        ####################################
                    else:
                        for key in model_param:
                            global_param[key]+=model_param[key]*fed_avg_weight[idx]
                        ####################################
                        # print(f'??????????????????{idx}???-model_param????????????????????????global_param')
                        # print(f'weight:{fed_avg_weight[idx]}')
                        # print(model_param['classifier.0.weight'])
                        # print(global_param['classifier.0.weight'])
                        ####################################


            global_model.load_state_dict(global_param)
            ###################################
            # print(f'??????????????????????????????global_param:')
            # print(global_param['classifier.0.weight'])
            # # print(f'??????????????????copy????????????global_param_copy:')
            # # print(global_param_copy['classifier.0.weight'])
            ###################################

            if args.save_rounds>0 and (round+1)%args.save_rounds==0:
                #save global model checkpoint
                """??????????????????"""
                fed_output_dir=os.path.join(args.fed_output_dir,'checkpoint--{}--{}'.format(args.algorithm,round))
                if not os.path.exists(fed_output_dir):
                    os.makedirs(fed_output_dir)
                model_to_save=global_model.module if hasattr(global_model,'module') else global_model
                model_to_save.save_pretrained(fed_output_dir)
                logger.info('Saving global model checkpoint to %s',fed_output_dir)

            logger.info(f'global num_test:{len(global_dataset)}')
            test_acc,test_f1=compute_metrics(global_model,test_dataloader_global,args.device)
            global_test_acc_dict[round]=test_acc

            logger.info(f'global model test acc:{test_acc}')
            logger.info(f'global model test f1:{test_f1}')

        #?????????????????????????????????
        avg_train_acc_file=os.path.join(args.fed_output_dir,'{}-{}-train_acc.txt'.format(
            args.algorithm,args.partition_strategy))
        with open(avg_train_acc_file,'w') as f:
            f.writelines('algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                args.algorithm,args.partition_strategy,args.num_parties,args.sample,args.fed_num_train_epochs,args.communication_rounds))
            f.writelines('\n')
            for key,value in avg_train_acc_dict.items():
                f.writelines('round:{}--avg_train_acc:{}'.format(key,value))
                f.writelines('\n')

        #?????????????????????????????????
        global_test_acc_file=os.path.join(args.fed_output_dir,'{}-{}-test_acc.txt'.format(
            args.algorithm,args.partition_strategy))
        with open(global_test_acc_file,'w') as f:
            f.writelines(
                'algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                    args.algorithm, args.partition_strategy, args.num_parties, args.sample, args.fed_num_train_epochs,
                    args.communication_rounds))
            f.writelines('\n')
            for key,value in global_test_acc_dict.items():
                f.writelines('round:{}--global_test_acc:{}'.format(key,value))
                f.writelines('\n')


    return test_acc




def alg_fedagg_time(args,tokenizer,model_init):
    avg_train_acc_dict={}
    global_test_acc_dict={}
    test_acc=0
    seed_everything(args.seed)
    party_dataset = partition[args.partition_strategy](args,tokenizer)
    global_dataset = load_and_cache_examples(args,tokenizer,data_type='test')
    test_sampler = SequentialSampler(global_dataset)
    test_dataloader_global = DataLoader(global_dataset,args.fed_eval_batch_size,
                                        sampler=test_sampler, collate_fn=collate_fn)

    if args.algorithm == 'fedagg_time':
        logger.info('Initializeing models')
        models = init_models(args, args.num_parties, model_init)
        global_models = init_models(args, 1, model_init)
        global_model=global_models[0]

        #??????????????????????????????????????????global_param_list???
        global_param_list = []
        #????????????????????????????????????????????????????????????total_num_data_list???
        total_num_data_list=[]

        global_param=global_model.state_dict()


        if args.is_same_initial:
            for model_id,model in models.items():
                model.load_state_dict(global_param)

        for round in range(args.communication_rounds):
            logger.info(f'In Communication Round:{round}')

            arr=np.arange(args.num_parties)
            np.random.shuffle(arr)
            selected=arr[:int(args.num_parties*args.sample)]
            selected = sorted(selected)

            global_param=global_model.state_dict()
            #?????????????????????????????????????????????global_param_list???
            global_param_list.append(copy.deepcopy(global_param))

            ###################################
            # print(f'?????????????????????????????????global_param???')
            # print(global_param['classifier.0.weight'])
            # print(f'??????????????????????????????global_param_list??????????????????:')
            # print(global_param_list[-1]['classifier.0.weight'])
            ###################################


            if round==0:
                if args.is_same_initial:
                    for idx in selected:
                        models[idx].load_state_dict(global_param)

            else:
                for idx in selected:
                    models[idx].load_state_dict(global_param)

            models_list,avg_train_acc=train_local_model(args,models,selected,party_dataset)
            avg_train_acc_dict[round]=avg_train_acc

            #??????????????????
            #1.??????????????????????????????????????????
            total_num_data=sum(len(party_dataset[i]) for i in selected)
            # total_num_data_list.append(total_num_data)
            if total_num_data!=0:
                #2.?????????????????????????????????party?????????????????????????????????/???????????????
                fed_avg_weight=[len(party_dataset[i])/total_num_data for i in selected]
                #3.??????????????????????????????????????????
                for idx in range(len(selected)):
                    model_param=models[selected[idx]].state_dict()
                    if idx==0:
                        for key in model_param:
                            global_param[key]=model_param[key]*fed_avg_weight[idx]
                    else:
                        for key in model_param:
                            global_param[key]+=model_param[key]*fed_avg_weight[idx]



                #?????????????????????????????????????????????????????????????????????????????????????????????
                global_param_list.append(copy.deepcopy(global_param))
                # print('???????????????????????????????????????????????????')
                # print(global_param['classifier.0.weight'])
                # for i in range(len(global_param_list)):
                #     print(f'???{i}??????????????????')
                #     print(global_param_list[i]['classifier.0.weight'])


                # print('*'*20)
                # for i in range(round+1):
                #     print(f'list--i:{i}')
                #     print(global_param_list[i]['classifier.0.weight'])

                #4.?????????????????????????????????????????????????????????????????????
                #total_num_data_list????????????????????????????????????????????????????????????????????????????????????????????????????????????
                # global_total_num_data=sum(total_num_data_list[max(round-args.acc_rounds+1,0):round+1])
                # global_fed_avg_num_weight = [np.round((total_num_data_list[i] / global_total_num_data), 3)
                #                          for i in range(max(round - args.acc_rounds + 1, 0), round + 1)]
                global_fed_avg_time_weight = [np.round(1 / np.exp(round - j), 3) for j in
                                             range(max(0, round - args.acc_rounds + 1), round + 1)]
                # print(f'global_weight:{global_fed_avg_time_weight}')
                sum_weight=sum(global_fed_avg_time_weight)

                # if round>10:
                #     for j in range(min(args.acc_rounds,round+1)):
                #         # current_num_weight=global_fed_avg_num_weight[j]
                #         current_time_weight=global_fed_avg_time_weight[j]
                #         if j==0:
                #             for key in global_param:
                #                 global_param[key]=current_time_weight\
                #                                   *global_param_list[max(0,round-args.acc_rounds+1+j)][key]
                #         else:
                #             for key in global_param:
                #                 global_param[key]+=current_time_weight\
                #                                    *global_param_list[max(0,round-args.acc_rounds+1+j)][key]
                if round-args.acc_rounds+1>0:
                    for j in range(args.acc_rounds):
                        #1.???????????????
                        current_time_weight=global_fed_avg_time_weight[j]/sum_weight
                        #2.??????????????????
                        # current_time_weight=global_fed_avg_time_weight[j]
                        if j==0:
                            for key in global_param:
                                global_param[key]=current_time_weight*\
                                                  global_param_list[round-args.acc_rounds+2+j][key]
                            ################################
                            # print(current_time_weight)
                            # print(global_param_list[round-args.acc_rounds+2+j]['classifier.0.weight'])
                            # print('?????????j=0???')
                            # print(global_param['classifier.0.weight'])
                            ################################
                        else:
                            for key in global_param:
                                global_param[key]=global_param[key]+current_time_weight*\
                                                   global_param_list[round-args.acc_rounds+2+j][key]
                            ################################
                            # print(current_time_weight)
                            # print(global_param_list[round-args.acc_rounds+2+j]['classifier.0.weight'])
                            # print(f'?????????j={j}???')
                            # print(global_param['classifier.0.weight'])
                            ################################

            # for i in range(round+1):
            #     print(f'list--i:{i}')
            #     print(global_param_list[i]['classifier.0.weight'])
            
            global_model.load_state_dict(global_param)
            ######################################
            # print(f'???????????????????????????global_param:')
            # print(global_param['classifier.0.weight'])
            #
            # print(f'???????????????????????????global_param??????????????????????????????????????????pop??????')
            # print(global_param_list[-1]['classifier.0.weight'])
            ######################################

            if args.save_rounds>0 and (round+1)%args.save_rounds==0:
                #save global model checkpoint
                """??????????????????"""
                fed_output_dir=os.path.join(args.fed_output_dir,'checkpoint--{}--{}'.format(args.algorithm,round))
                if not os.path.exists(fed_output_dir):
                    os.makedirs(fed_output_dir)
                model_to_save=global_model.module if hasattr(global_model,'module') else global_model
                model_to_save.save_pretrained(fed_output_dir)
                logger.info('Saving global model checkpoint to %s',fed_output_dir)

            #???global_param_list??????????????????????????????????????????????????????
            global_param_list.pop()

            logger.info(f'global num_test:{len(global_dataset)}')
            test_acc,test_f1=compute_metrics(global_model,test_dataloader_global,args.device)
            global_test_acc_dict[round]=test_acc

            logger.info(f'global model test acc:{test_acc}')
            logger.info(f'global model test f1:{test_f1}')

        #?????????????????????????????????
        avg_train_acc_file=os.path.join(args.fed_output_dir,'{}-{}-train_acc.txt'.format(
            args.algorithm,args.partition_strategy))
        with open(avg_train_acc_file,'w') as f:
            f.writelines('algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                args.algorithm,args.partition_strategy,args.num_parties,args.sample,args.fed_num_train_epochs,args.communication_rounds))
            f.writelines('\n')
            for key,value in avg_train_acc_dict.items():
                f.writelines('round:{}--avg_train_acc:{}'.format(key,value))
                f.writelines('\n')

        #?????????????????????????????????
        global_test_acc_file=os.path.join(args.fed_output_dir,'{}-{}-test_acc.txt'.format(
            args.algorithm,args.partition_strategy))
        with open(global_test_acc_file,'w') as f:
            f.writelines(
                'algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                    args.algorithm, args.partition_strategy, args.num_parties, args.sample, args.fed_num_train_epochs,
                    args.communication_rounds))
            f.writelines('\n')
            for key,value in global_test_acc_dict.items():
                f.writelines('round:{}--global_test_acc:{}'.format(key,value))
                f.writelines('\n')
    return test_acc


def alg_fedagg_prox(args, tokenizer, model_init):
    avg_train_acc_dict = {}
    global_test_acc_dict = {}
    test_acc = 0
    seed_everything(args.seed)
    party_dataset = partition[args.partition_strategy](args, tokenizer)
    global_dataset = load_and_cache_examples(args, tokenizer, data_type='test')
    test_sampler = SequentialSampler(global_dataset)
    test_dataloader_global = DataLoader(global_dataset, args.fed_eval_batch_size,
                                        sampler=test_sampler, collate_fn=collate_fn)

    if args.algorithm == 'fedagg_prox':
        logger.info('Initializeing models')
        models = init_models(args, args.num_parties, model_init)
        global_models = init_models(args, 1, model_init)
        global_model = global_models[0]

        # ??????????????????????????????????????????global_param_list???
        global_param_list = []
        # ????????????????????????????????????????????????????????????total_num_data_list???
        total_num_data_list = []

        global_param = global_model.state_dict()

        if args.is_same_initial:
            for model_id, model in models.items():
                model.load_state_dict(global_param)

        for round in range(args.communication_rounds):
            logger.info(f'In Communication Round:{round}')

            arr = np.arange(args.num_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.num_parties * args.sample)]
            selected = sorted(selected)

            global_param = global_model.state_dict()
            # ?????????????????????????????????????????????global_param_list???
            global_param_list.append(copy.deepcopy(global_param))

            ###################################
            # print(f'?????????????????????????????????global_param???')
            # print(global_param['classifier.0.weight'])
            # print(f'??????????????????????????????global_param_list??????????????????:')
            # print(global_param_list[-1]['classifier.0.weight'])
            ###################################

            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        models[idx].load_state_dict(global_param)

            else:
                for idx in selected:
                    models[idx].load_state_dict(global_param)

            models_list, avg_train_acc = train_local_fedprox_model(models,global_model,selected,party_dataset,args)
            avg_train_acc_dict[round] = avg_train_acc

            # ??????????????????
            # 1.??????????????????????????????????????????
            total_num_data = sum(len(party_dataset[i]) for i in selected)
            # total_num_data_list.append(total_num_data)
            if total_num_data!=0:
                # 2.?????????????????????????????????party?????????????????????????????????/???????????????
                fed_avg_weight = [len(party_dataset[i]) / total_num_data for i in selected]
                # 3.??????????????????????????????????????????
                for idx in range(len(selected)):
                    model_param = models[selected[idx]].state_dict()
                    if idx == 0:
                        for key in model_param:
                            global_param[key] = model_param[key] * fed_avg_weight[idx]
                    else:
                        for key in model_param:
                            global_param[key] += model_param[key] * fed_avg_weight[idx]

                # ?????????????????????????????????????????????????????????????????????????????????????????????
                global_param_list.append(copy.deepcopy(global_param))
                # print('???????????????????????????????????????????????????')
                # print(global_param['classifier.0.weight'])
                # for i in range(len(global_param_list)):
                #     print(f'???{i}??????????????????')
                #     print(global_param_list[i]['classifier.0.weight'])

                # print('*'*20)
                # for i in range(round+1):
                #     print(f'list--i:{i}')
                #     print(global_param_list[i]['classifier.0.weight'])

                # 4.?????????????????????????????????????????????????????????????????????
                # total_num_data_list????????????????????????????????????????????????????????????????????????????????????????????????????????????
                # global_total_num_data=sum(total_num_data_list[max(round-args.acc_rounds+1,0):round+1])
                # global_fed_avg_num_weight = [np.round((total_num_data_list[i] / global_total_num_data), 3)
                #                          for i in range(max(round - args.acc_rounds + 1, 0), round + 1)]
                global_fed_avg_time_weight = [np.round(1 / np.exp(round - j), 3) for j in
                                              range(max(0, round - args.acc_rounds + 1), round + 1)]
                # print(f'global_weight:{global_fed_avg_time_weight}')
                sum_weight = sum(global_fed_avg_time_weight)

                # if round>10:
                #     for j in range(min(args.acc_rounds,round+1)):
                #         # current_num_weight=global_fed_avg_num_weight[j]
                #         current_time_weight=global_fed_avg_time_weight[j]
                #         if j==0:
                #             for key in global_param:
                #                 global_param[key]=current_time_weight\
                #                                   *global_param_list[max(0,round-args.acc_rounds+1+j)][key]
                #         else:
                #             for key in global_param:
                #                 global_param[key]+=current_time_weight\
                #                                    *global_param_list[max(0,round-args.acc_rounds+1+j)][key]
                if round - args.acc_rounds + 1 > 0:
                    for j in range(args.acc_rounds):
                        current_time_weight = global_fed_avg_time_weight[j] / sum_weight
                        if j == 0:
                            for key in global_param:
                                global_param[key] = current_time_weight * \
                                                    global_param_list[round - args.acc_rounds + 2 + j][key]
                            ################################
                            # print(current_time_weight)
                            # print(global_param_list[round-args.acc_rounds+2+j]['classifier.0.weight'])
                            # print('?????????j=0???')
                            # print(global_param['classifier.0.weight'])
                            ################################
                        else:
                            for key in global_param:
                                global_param[key] = global_param[key] + current_time_weight * \
                                                    global_param_list[round - args.acc_rounds + 2 + j][key]
                            ################################
                            # print(current_time_weight)
                            # print(global_param_list[round-args.acc_rounds+2+j]['classifier.0.weight'])
                            # print(f'?????????j={j}???')
                            # print(global_param['classifier.0.weight'])
                            ################################

            # for i in range(round+1):
            #     print(f'list--i:{i}')
            #     print(global_param_list[i]['classifier.0.weight'])

            global_model.load_state_dict(global_param)
            ######################################
            # print(f'???????????????????????????global_param:')
            # print(global_param['classifier.0.weight'])
            #
            # print(f'???????????????????????????global_param??????????????????????????????????????????pop??????')
            # print(global_param_list[-1]['classifier.0.weight'])
            ######################################

            if args.save_rounds > 0 and (round + 1) % args.save_rounds == 0:
                # save global model checkpoint
                """??????????????????"""
                fed_output_dir = os.path.join(args.fed_output_dir, 'checkpoint--{}--{}'.format(args.algorithm, round))
                if not os.path.exists(fed_output_dir):
                    os.makedirs(fed_output_dir)
                model_to_save = global_model.module if hasattr(global_model, 'module') else global_model
                model_to_save.save_pretrained(fed_output_dir)
                logger.info('Saving global model checkpoint to %s', fed_output_dir)

            # ???global_param_list??????????????????????????????????????????????????????
            global_param_list.pop()

            logger.info(f'global num_test:{len(global_dataset)}')
            test_acc, test_f1 = compute_metrics(global_model, test_dataloader_global, args.device)
            global_test_acc_dict[round] = test_acc

            logger.info(f'global model test acc:{test_acc}')
            logger.info(f'global model test f1:{test_f1}')

        # ?????????????????????????????????
        avg_train_acc_file = os.path.join(args.fed_output_dir, '{}-{}-train_acc.txt'.format(
            args.algorithm, args.partition_strategy))
        with open(avg_train_acc_file, 'w') as f:
            f.writelines(
                'algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                    args.algorithm, args.partition_strategy, args.num_parties, args.sample, args.fed_num_train_epochs,
                    args.communication_rounds))
            f.writelines('\n')
            for key, value in avg_train_acc_dict.items():
                f.writelines('round:{}--avg_train_acc:{}'.format(key, value))
                f.writelines('\n')

        # ?????????????????????????????????
        global_test_acc_file = os.path.join(args.fed_output_dir, '{}-{}-test_acc.txt'.format(
            args.algorithm, args.partition_strategy))
        with open(global_test_acc_file, 'w') as f:
            f.writelines(
                'algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                    args.algorithm, args.partition_strategy, args.num_parties, args.sample, args.fed_num_train_epochs,
                    args.communication_rounds))
            f.writelines('\n')
            for key, value in global_test_acc_dict.items():
                f.writelines('round:{}--global_test_acc:{}'.format(key, value))
                f.writelines('\n')
    return test_acc





"""3.??????fednova?????????????????????"""
def train_fednova_model(model_id, global_model, train_dataloader, model, args):
    """ Train the model """
    logger.info('Training network %s' % str(model_id))

    server_model = copy.deepcopy(global_model).to(args.device)
    server_param=copy.deepcopy(server_model.state_dict())

    ##########################################
    # print(f'????????????global_param:')
    # print(global_model.state_dict()['classifier.0.weight'])
    # print(f'????????????server_param:')
    # print(server_model.state_dict()['classifier.0.weight'])
    ##########################################

    classifier_params = ['classifier']
    optimizer_group_parameters = [
        # 1.albert???????????????????????????
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': args.classifier_lr},
        # 2.????????????????????????
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in classifier_params)],
         'lr': args.fed_lr}
    ]
    optimizer = AdamW(params=optimizer_group_parameters, eps=args.fed_adam_epsilon)
    # optimizer=torch.optim.SGD(params=optimizer_group_parameters,momentum=0.9)
    global_step = 0
    tr_loss = 0.0

    # ???????????????????????????????????????????????????
    # global_weight_collector=list(global_model.to(args.device).parameters())
    # global_params=global_model.state_dict()
    model.to(args.device)
    model.train()
    # model.zero_grad()
    print(f'Training network party:{model_id}')
    tao=0.0
    for epoch in range(int(args.fed_num_train_epochs)):
        # ????????????epoch???loss
        epoch_loss_collector = []
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(),args.fed_max_grad_norm)
            tr_loss += loss.item()
            optimizer.step()
            # model.zero_grad()
            optimizer.zero_grad()
            global_step += 1
            epoch_loss_collector.append(loss.item())
            tao+=1
            pbar(step, {'loss': loss.item()})
        print(" ")
        # ??????epoch???????????????
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info(f'Epoch:{epoch} Loss:{epoch_loss}')
    # print('********************')
    # print(f'model:{model_id}')
    # print(model.state_dict()['classifier.0.weight'])

    model_param=model.state_dict()
    delta_param=copy.deepcopy(server_param)
    for key in delta_param:
        #delta_param???????????????????????????/tao???
        delta_param[key]=torch.true_divide(server_param[key]-model_param[key],tao)

    train_acc, train_f1 = compute_metrics(model, train_dataloader, args.device)


    logger.info(f'Training accuracy:{train_acc}')
    logger.info(f'Training f1 score:{train_f1}')

    logger.info(f'Global Train Loss:{tr_loss / global_step}')

    logger.info('***Traning complete***')
    return train_acc, train_f1, (tr_loss / global_step),tao,delta_param


"""????????????party????????????fednova?????????????????????"""
def train_local_fednova_model(models, global_model, selected, party_dataset, args):
    avg_train_acc = 0.0
    non_zero_party = 0

    #???selected???client?????????tao(????????????),delta_param(client??????????????????)???????????????????????????????????????
    tao_list=[]
    delta_param_list=[]
    num_data_list=[]

    for model_id, model in models.items():
        if model_id not in selected:
            continue
        # print('??????????????????????????????')
        # print(models[17].state_dict()['classifier.0.weight'])
        # print('??????????????????????????????')
        # print(models[model_id].state_dict()['classifier.0.weight'])
        # ?????????party??????????????????

        if len(party_dataset[model_id]) > 0:
            non_zero_party += 1
            local_dataset = party_dataset[model_id]
            logger.info(f'Training model:{model_id},n_training:{len(local_dataset)}')
            model.to(args.device)

            # party???train_dataloader
            local_dataset = load_dataset(local_dataset)
            train_sampler = RandomSampler(local_dataset)
            train_dataloader_local = DataLoader(local_dataset, args.fed_train_batch_size,
                                                sampler=train_sampler, collate_fn=collate_fn)

            train_acc, train_f1, train_avg_loss,tao,delta_param = train_fednova_model(model_id, global_model, train_dataloader_local,
                                                                      model, args)
            avg_train_acc += train_acc

            tao_list.append(tao)
            delta_param_list.append(copy.deepcopy(delta_param))
            num_data_list.append(len(local_dataset))
        # models[model_id]=model
        # print('?????????????????????????????????')
        # print(models[17].state_dict()['classifier.0.weight'])
        # print('???????????????????????????')
        # print(models[model_id].state_dict()['classifier.0.weight'])

    if non_zero_party != 0:
        avg_train_acc /= non_zero_party
    else:
        avg_train_acc = 0

    if args.algorithm == 'local_training':
        logger.info(f'avg train acc:{avg_train_acc}')

    models_list = list(models.values())
    return models_list, avg_train_acc,tao_list,delta_param_list,num_data_list

"""??????fednova??????"""
def alg_fednova(args, tokenizer, model_init):
    avg_train_acc_dict = {}
    global_test_acc_dict = {}
    test_acc = 0
    seed_everything(args.seed)
    party_dataset = partition[args.partition_strategy](args, tokenizer)
    global_dataset = load_and_cache_examples(args, tokenizer, data_type='test')
    test_sampler = SequentialSampler(global_dataset)
    test_dataloader_global = DataLoader(global_dataset, args.fed_eval_batch_size,
                                        sampler=test_sampler, collate_fn=collate_fn)

    if args.algorithm == 'fednova':
        logger.info('Initializeing models')
        models = init_models(args, args.num_parties, model_init)
        global_models = init_models(args, 1, model_init)
        global_model = global_models[0]


        global_param = global_model.state_dict()

        if args.is_same_initial:
            for model_id, model in models.items():
                model.load_state_dict(global_param)

        for round in range(args.communication_rounds):
            logger.info(f'In Communication Round:{round}')

            arr = np.arange(args.num_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.num_parties * args.sample)]
            selected = sorted(selected)

            global_param = global_model.state_dict()

            ###################################
            # print(f'?????????????????????????????????global_param???')
            # print(global_param['classifier.0.weight'])
            # print(f'??????????????????????????????global_param_list??????????????????:')
            # print(global_param_list[-1]['classifier.0.weight'])
            ###################################

            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        models[idx].load_state_dict(global_param)

            else:
                for idx in selected:
                    models[idx].load_state_dict(global_param)

            models_list, avg_train_acc,tao_list,delta_param_list,num_data_list =train_local_fednova_model(models, global_model, selected, party_dataset, args)
            avg_train_acc_dict[round] = avg_train_acc

            non_zero_selected=0
            for i in range(len(selected)):
                if len(party_dataset[selected[i]])>0:
                    non_zero_selected+=1
            #
            # print(f'non_zero_selected:{non_zero_selected}')
            # print(f'tao_list:{len(tao_list)}')
            # print(f'delta_param_list:{len(delta_param_list)}')
            # print(f'num_data_list:{len(num_data_list)}')

            #?????????????????????????????????n
            total_num_data=sum(num_data_list)

            server_model = copy.deepcopy(global_model)
            update_param = copy.deepcopy(server_model.to(args.device).state_dict())
            # print(update_param[key].device)
            # print(delta_round[key].device)

            if total_num_data!=0:
                #copy???????????????????????????????????????fednova???????????????????????????(Di/n)/(taoi)*delta
                delta_round=copy.deepcopy(global_model.state_dict())
                for key in delta_round:
                    delta_round[key]=0.0


                for i in range(non_zero_selected):
                    delta_param=delta_param_list[i]
                    for key in delta_round:
                        delta_round[key]+=delta_param[key]*num_data_list[i]/total_num_data

                #??????????????????????????????
                coeff=0.0
                for i in range(non_zero_selected):
                    coeff+=tao_list[i]*num_data_list[i]/total_num_data


                for key in update_param:
                    update_param[key]-=coeff*delta_round[key]

            global_model.load_state_dict(update_param)


            ######################################
            # print(f'???????????????????????????global_param:')
            # print(global_param['classifier.0.weight'])
            #
            # print(f'???????????????????????????global_param??????????????????????????????????????????pop??????')
            # print(global_param_list[-1]['classifier.0.weight'])
            ######################################

            if args.save_rounds > 0 and (round + 1) % args.save_rounds == 0:
                # save global model checkpoint
                """??????????????????"""
                fed_output_dir = os.path.join(args.fed_output_dir, 'checkpoint--{}--{}'.format(args.algorithm, round))
                if not os.path.exists(fed_output_dir):
                    os.makedirs(fed_output_dir)
                model_to_save = global_model.module if hasattr(global_model, 'module') else global_model
                model_to_save.save_pretrained(fed_output_dir)
                logger.info('Saving global model checkpoint to %s', fed_output_dir)


            logger.info(f'global num_test:{len(global_dataset)}')
            test_acc, test_f1 = compute_metrics(global_model, test_dataloader_global, args.device)
            global_test_acc_dict[round] = test_acc

            logger.info(f'global model test acc:{test_acc}')
            logger.info(f'global model test f1:{test_f1}')

        # ?????????????????????????????????
        avg_train_acc_file = os.path.join(args.fed_output_dir, '{}-{}-train_acc.txt'.format(
            args.algorithm, args.partition_strategy))
        with open(avg_train_acc_file, 'w') as f:
            f.writelines(
                'algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                    args.algorithm, args.partition_strategy, args.num_parties, args.sample, args.fed_num_train_epochs,
                    args.communication_rounds))
            f.writelines('\n')
            for key, value in avg_train_acc_dict.items():
                f.writelines('round:{}--avg_train_acc:{}'.format(key, value))
                f.writelines('\n')

        # ?????????????????????????????????
        global_test_acc_file = os.path.join(args.fed_output_dir, '{}-{}-test_acc.txt'.format(
            args.algorithm, args.partition_strategy))
        with open(global_test_acc_file, 'w') as f:
            f.writelines(
                'algorithm:{},partition_strategy:{},num_parties:{},sample:{},local_epochs:{},communication_rounds:{}'.format(
                    args.algorithm, args.partition_strategy, args.num_parties, args.sample, args.fed_num_train_epochs,
                    args.communication_rounds))
            f.writelines('\n')
            for key, value in global_test_acc_dict.items():
                f.writelines('round:{}--global_test_acc:{}'.format(key, value))
                f.writelines('\n')
    return test_acc




def main():
    test_acc=0
    parser=argparse.ArgumentParser()
    #albert_model???????????????
    parser.add_argument('--model_name_or_path',default=None,type=str,required=True,
                        help='Path to pre-trained model or shortcut name selected in the list')
    parser.add_argument('--task_name',default=None,type=str,required=True,
                        help='The name of the task to train selected in the list:"+"'.join(processors.keys()))
    parser.add_argument('--data_dir',default=None,type=str,required=True,
                        help='The input data dir. Should contain the .tsv files (or other data files) for the task')
    parser.add_argument('--fed_output_dir',default=None,type=str,required=True,
                        help='The output directory where the model predictions and checkpoints will be written')
    parser.add_argument('--vocab_file',default='',type=str,required=True,
                        help='The path to vocab.txt of the pre-trained model')
    parser.add_argument('--model_type',default=None,type=str,required=True,
                        help='Model type')

    parser.add_argument('--do_lower_case',action='store_true',
                        help='Set the flag if you are using an uncased model')
    parser.add_argument('--max_seq_length',default=100,type=int,
                        help='The maximum total input sequence length after tokenization. Sequences longer than'
                             'this will be truncated, otherwise be padded')
    parser.add_argument('--local_rank',default=-1,type=int,
                        help='For distributed training:local_rank')
    # parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu',type=str,
    #                     help='The device to run the programme')

    parser.add_argument('--fed_lr',default=5e-5,type=float,
                        help='The initial learning rate of the pre-trained layers for Adam under fed setting')
    parser.add_argument('--classifier_lr',default=1e-3,type=float,
                        help='The initial learning rate of the classifier layers for Adam under fed setting')
    parser.add_argument('--fed_adam_epsilon',default=1e-6, type=float,
                        help='Epsilon for Adam optimizer')
    parser.add_argument('--fed_num_train_epochs',default=5,type=int,
                        help='Number of local epochs for each selected party under fed setting')
    parser.add_argument('--fed_max_grad_norm',default=1.0,type=float,
                        help='Max gradient norm')
    parser.add_argument('--fed_train_batch_size',default=32,type=int,
                        help='Local training batch size for each selected party under fed setting')
    parser.add_argument('--fed_eval_batch_size',default=64,type=int,
                        help='Global evaling batch size under fed setting ')
    parser.add_argument('--algorithm',default=None,type=str,required=True,
                        help='The federated learning algorithms in the list')
    parser.add_argument('--seed',default=42,type=int,
                        help='Random seed for initialization')
    parser.add_argument('--num_parties',default=20,type=int,
                        help='Number of total parties in the fed setting')
    parser.add_argument('--sample',default=0.2,type=float,
                        help='Sample ration for each communication round')
    parser.add_argument('--communication_rounds',default=50,type=int,
                        help='Number of maximum communication rounds under fed setting')
    parser.add_argument('--is_same_initial',default=True,type=bool,
                        help='Whether initial all the models with the same params')
    parser.add_argument('--partition_strategy',default=None,type=str,required=True,
                        help='The data partitioning strategy')
    parser.add_argument('--beta',default=100,type=float,
                        help='the param control the shrew of the dirichlet distribution')
    parser.add_argument('--alpha',default=0.5,type=float,
                        help='the weight of current model in the fedagg_simple algorithm')
    parser.add_argument('--acc_rounds',default=4,type=int,
                        help='the total global models used for the params aggregation')
    parser.add_argument('--mu',default=1,type=float,
                        help='the mu param for fedprox algorithm contorls the regularization')

    parser.add_argument('--save_rounds',default=10,type=int,
                        help='save checkpoint ever X updates rounds')
    parser.add_argument('--k',default=1,type=int,
                        help='control the non_iid_label_k,the type of labels one party owns')



    args=parser.parse_args()

    if not os.path.exists(args.fed_output_dir):
        os.makedirs(args.fed_output_dir)
    args.fed_output_dir=args.fed_output_dir+'{}'.format(args.model_type)
    if not os.path.exists(args.fed_output_dir):
        os.makedirs(args.fed_output_dir)

    init_logger(log_file=args.fed_output_dir+'/--{}.log'.format(args.task_name))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.device=device
    seed_everything(args.seed)

    args.task_name=args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError('Task not found:{}'.format(args.task_name))

    processor=processors[args.task_name]()
    label_list=processor.get_labels()
    num_labels=len(label_list)


    args.model_type=args.model_type.lower()
    config = AlbertConfig.from_pretrained(args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name)
    tokenizer = FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    model= AlbertMyClassifier.from_pretrained(args.model_name_or_path,
                                               from_tf=bool('.ckpt' in args.model_name_or_path),
                                               config=config)

    if args.algorithm=='fedavg':
        logger.info('Algorithm---Federated Average')
        test_acc=alg_fedavg(args,tokenizer=tokenizer,model_init=model)

    elif args.algorithm=='fedprox':
        logger.info('Algorithm---Federated Prox')
        test_acc=alg_fedprox(args,tokenizer=tokenizer,model_init=model)

    # elif args.algorithm=='fedagg_simple':
    #     logger.info('Algorithm---Federated Aggregation Simple Version')
    #     test_acc=alg_fedagg_simple(args,tokenizer=tokenizer,model_init=model_init)

    # elif args.algorithm=='fedagg_num_time':
    #     logger.info('Algorithm---Federated Aggregation Time Version')
    #     test_acc=alg_fedagg_num_time(args,tokenizer=tokenizer,model_init=model)
    
    elif args.algorithm=='fedagg_time':
        logger.info('Algorithm---Federated Aggregation Time Version Another')
        test_acc=alg_fedagg_time(args,tokenizer=tokenizer,model_init=model)

    elif args.algorithm=='fedagg_prox':
        logger.info('Algorithm---Federated Aggregation And FedProx')
        test_acc=alg_fedagg_prox(args,tokenizer=tokenizer,model_init=model)

    elif args.algorithm=='fednova':
        logger.info('Algorithm---FedNova')
        test_acc=alg_fednova(args,tokenizer=tokenizer,model_init=model)

    # elif args.algorithm=='none':
    #     logger.info('Centralized Training')
    #     test_acc=non_fed(args,tokenizer=tokenizer,model_init=model)

    else:
        raise ValueError('Algorithm not found:{}'.format(args.algorithm))
    return test_acc


if __name__=="__main__":
    main()



