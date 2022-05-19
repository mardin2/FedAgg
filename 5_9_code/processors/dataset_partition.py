"""
根据不同的数据划分策略和客户端的数目划分global数据集
"""
from __future__ import absolute_import, division, print_function
import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, RandomSampler,TensorDataset
from processors import glue_output_modes as output_modes, collate_fn
from processors import glue_processors as processors
from processors import glue_convert_examples_to_features as convert_examples_to_features
from tools.common import logger
from model.tokenization_albert import FullTokenizer
from model.configuration_albert import AlbertConfig


"""读取数据集文件---返回数据集([0]:all_input_id,[1]all_attention_mask,[2]all_token_type_id,[3]all_len,[4]labels"""
def load_and_cache_examples(args,tokenizer,data_type='train'):

    processor = processors[args.task_name]()
    output_mode = output_modes[args.task_name]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s",args.data_dir)
        label_list = processor.get_labels()

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                output_mode = output_mode)
        # print(f'内部：{features[0]}')
        if args.local_rank in [-1, 0]:
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

"""加载全局数据集"""
# processor=processors['tnews']()
# model_name_or_path='../prev_trained_model/albert_small_zh'
# vocab_file='../prev_trained_model/albert_small_zh/vocab.txt'
# do_lower_case=True
# data_dir='../dataset/tnews'
# max_sequence_length=50
# local_rank=-1
# num_label=len(processor.get_labels())
# config = AlbertConfig.from_pretrained(model_name_or_path,
#                                       num_labels=num_label,
#                                       finetuning_task='tnews')
# tokenizer =FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
#
# global_dataset=load_and_cache_examples('tnews',tokenizer,data_dir,model_name_or_path,max_sequence_length,
#                                        local_rank)
#
# label_list=processor.get_labels()
# examples=processor.get_train_examples(data_dir)
# features=convert_examples_to_features(examples,tokenizer,max_seq_length=max_sequence_length,task='tnews',label_list=label_list,
#                                       output_mode='classification')


# print(f'外部：{features[0]}')


"""
获取所有样本对应的特征
"""
def get_all_label_features(args,tokenizer,data_type='train'):
    processor = processors[args.task_name]()
    # 所有的样本序列
    examples = processor.get_train_examples(args.data_dir)
    # 任务对应的输出形式
    output_mode = output_modes[args.task_name]
    # 映射前的原始标签序列
    label_list = processor.get_labels()
    # 映射后的标签序列
    label_list_map = processor.get_label_map()
    # 标签的个数
    num_labels = len(label_list_map)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s",args.data_dir)
        # 所有的样本对应的特征序列（五个特征）
        features = convert_examples_to_features(examples, tokenizer, args.max_seq_length,
                                                args.task_name, label_list, output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    num_features=len(features)
    return features,num_labels,num_features



"""
得到各个label旗下的样本特征
"""
def get_label_features(args,tokenizer,data_type="train"):
    """
    得到各个label下的样本有哪些
    :return: list of label,
             list of label[i]:第i个label下的样本特征序列
             list of label[i][j]:第i个label下的的j样本对应的特征(input_id,attention_mask,token_type_id,
                                            label,input_id)
    """
    processor=processors[args.task_name]()
    # 所有的样本序列
    examples=processor.get_train_examples(args.data_dir)
    #任务对应的输出形式
    output_mode=output_modes[args.task_name]
    #映射前的原始标签序列
    label_list=processor.get_labels()
    #映射后的标签序列
    label_list_map=processor.get_label_map()
    #标签的个数
    num_labels=len(label_list_map)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s",args.data_dir)
    #所有的样本对应的特征序列（五个特征）
        features=convert_examples_to_features(examples,tokenizer,args.max_seq_length,args.task_name,label_list,output_mode)
    num_features=len(features)
    #Load label data features from cache or dataset file
    cached_label_features_file=os.path.join(args.data_dir,'cached_label_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(args.task_name)))
    if os.path.exists(cached_label_features_file):
        logger.info("Loading label features from cache file %s",cached_label_features_file)
        label_example_features=torch.load(cached_label_features_file)
    else:
        logger.info("Creating label features from data file at %s",args.data_dir)
        #各个label下的样本特征序列
        label_example_features=[[] for _ in range(num_labels)]
        #判断特征序列中的每个样本属于哪些label下
        for f in features:
            for i in range(num_labels):
                if f.label==label_list_map[i]:
                    label_example_features[i].append(f)
        if args.local_rank in [-1, 0]:
            logger.info("Saving label features into cached file %s", cached_label_features_file)
            torch.save(label_example_features, cached_label_features_file)
    return label_example_features,num_labels,num_features



# data_dir='../dataset/tnews'
# task='tnews'
# config = AlbertConfig.from_pretrained(model_name_or_path,
#                                       snum_labels=num_label,
#                                       finetuning_task='tnews')
# num_parties=10
# max_seq_length=50





"""
central环境：用全部训练数据在global端进行训练
"""
# def central(data_dir,task,tokenizer,max_seq_length,local_rank,model_name_or_path):
#     party_dataset=load_and_cache_examples(task,tokenizer,data_dir,model_name_or_path,max_seq_length,local_rank)
#     #返回已经加载到tensordataset中的数据集
#     return party_dataset



"""
1.独立同分布：随机取样(控制每个party得到的样本数尽可能相同)
(1)生成样本长度的索引序列idxs
(2)随机打乱样本序列
(3)根据客户端num_parties数目划分(2)中的样本序列batch_idxs
(4)根据(3)中得到的每个客户端的样本索引，找到对应样本
"""
def iid_quantity(args,tokenizer):
    features,num_lables,num_features=get_all_label_features(args,tokenizer)
    #确定样本总数
    num_global_train_data =num_features
    # 2.生成全局训练样本的索引并随机打乱
    idxs = np.random.permutation(num_global_train_data)
    # 3.根据客户端数划分全局训练样本索引
    batch_idxs = np.array_split(idxs, args.num_parties)
    # 4.根据batch索引取出全局训练样本中对应的样本
    party_dataset = [[] for _ in range(args.num_parties)]
    for i in range(args.num_parties):
        party_dataset[i].extend(np.array(features)[batch_idxs[i]])
    return party_dataset

# party_dataset=iid_quantity(num_parties,data_dir,task,tokenizer,max_seq_length,local_rank)
# for i in range(num_parties):
#     print(f'party{i}:{len(party_dataset[i])}')
#     print(party_dataset[i])
#     break


"""
2.独立同分布：(控制每个party得到的标签分布尽可能相同)
"""
def iid_label(args,tokenizer):
    party_dataset=[[]for _ in range(args.num_parties)]
    #各个label旗下的样本特征序列
    label_example_features,num_labels,num_features=get_label_features(args,tokenizer)
    # print(f'label_example_features:{len(label_example_features)}')
    #将各个label旗下的样本均匀地分配给每个party
    for i in range(num_labels):
        #该label旗下的样本特征序列
        current_label_example_features=np.array(label_example_features[i])
        #获取该label下的样本数量
        current_label_example_features_len=len(current_label_example_features)
        #随机打乱标签索引
        idxs=np.random.permutation(current_label_example_features_len)
        #根据客户端数量划分该label的索引
        batch_idxs=np.array_split(idxs,args.num_parties)
        for j in range(args.num_parties):
            #不将被追加的列表或元组当成一个整体，而是只追加列表中的元素，则可使用列表提供的 extend() 方法
            party_dataset[j].extend(current_label_example_features[batch_idxs[j]])
    return party_dataset
# party_dataset=iid_label(num_parties,num_label,data_dir,task, tokenizer, max_seq_length,local_rank)
# for i in range(num_parties):
#     print(len(party_dataset[i]))
#     break


"""
3.标签分布偏移(尽可能控制不存在数量分布偏移)
"""
# def non_iid_label(args,tokeizer):
#   party_dataset=[[]for _ in range(args.num_parties)]
#   # 各个label旗下的样本特征序列
#   label_example_features,num_labels,num_features = get_label_features(args,tokeizer)
#   mean_example=int(num_labels/args.num_parties)
#   min_length,max_length=0,5000
#   a,b=mean_example-0.5*mean_example,mean_example+0.5*mean_example
#   """
#   beta用于控制标签分布偏移的程度，beta越小，分布偏移程度越大
#   """
#   while min_length<a or max_length>b:
#     party_dataset=[[]for _ in range(args.num_parties)]
#     all_batch_idxs=[[] for _ in range(args.num_parties)]
#     #将各个label旗下的样本按照狄利克雷分布分配给每个party
#     for i in range(num_labels):
#         #该label旗下的样本特征序列
#         current_label_example_features=np.array(label_example_features[i])
#         #获取该label下的样本数量
#         current_label_example_features_len=len(current_label_example_features)
#         #生成样本索引
#         idxs=np.random.permutation(current_label_example_features_len)
#         #生成狄利克雷分布
#         proportions=np.random.dirichlet(np.repeat(args.beta,args.num_parties))
#         proportions=proportions/proportions.sum()
#         proportions = (np.cumsum(proportions) * current_label_example_features_len).astype(int)[:-1]
#         #根据狄利克雷分布划分该label的索引
#         batch_idxs=np.split(idxs,proportions)
#         # print(f'label{i}:{batch_idxs}')
#         # print('*'*20)
#         for j in range(args.num_parties):
#             #不将被追加的列表或元组当成一个整体，而是只追加列表中的元素，则可使用列表提供的 extend() 方法
#             party_dataset[j].extend(current_label_example_features[batch_idxs[j]])
#         for j in range(args.num_parties):
#             all_batch_idxs[j].extend(batch_idxs[j])
#     min_length=min(len(party_label) for party_label in all_batch_idxs)
#     max_length=max(len(party_label) for party_label in all_batch_idxs)
#   # print(min_length)
#   # print(max_length)
#   return party_dataset


# party_dataset=non_iid_label(num_parties,num_label,data_dir,task,
#                             tokenizer, max_seq_length,local_rank,num_features=2000,beta=100)

# for i in range(num_parties):
#     print(f'party{i}:{len(party_dataset[i])}')


def non_iid_label(args,tokeizer):
  #for online shopping,对于tnew去掉max和min_length
    max_length=10000
    # min_length=0
    # 各个label旗下的样本特征序列
    label_example_features,num_labels,num_features = get_label_features(args,tokeizer)

    #将各个label旗下的样本按照狄利克雷分布分配给每个party
    while max_length>4000:
        party_dataset = [[] for _ in range(args.num_parties)]
        all_batch_idxs=[[] for _ in range(args.num_parties)]
        for i in range(num_labels):
            #该label旗下的样本特征序列
            current_label_example_features=np.array(label_example_features[i])
            #获取该label下的样本数量
            current_label_example_features_len=len(current_label_example_features)
            #生成样本索引
            idxs=np.random.permutation(current_label_example_features_len)
            #生成狄利克雷分布
            proportions=np.random.dirichlet(np.repeat(args.beta,args.num_parties))
            proportions=proportions/proportions.sum()
            proportions = (np.cumsum(proportions) * current_label_example_features_len).astype(int)[:-1]
            #根据狄利克雷分布划分该label的索引
            batch_idxs=np.split(idxs,proportions)
            # print(f'label{i}:{batch_idxs}')
            # print('*'*20)
            for j in range(args.num_parties):
                #不将被追加的列表或元组当成一个整体，而是只追加列表中的元素，则可使用列表提供的 extend() 方法
                party_dataset[j].extend(current_label_example_features[batch_idxs[j]])
            for j in range(args.num_parties):
                all_batch_idxs[j].extend(batch_idxs[j])
            max_length=max(len(party_label) for party_label in all_batch_idxs)
            # min_length=min(len(party_label) for party_label in all_batch_idxs)           
    return party_dataset


def non_iid_label_k(args,tokenizer):
    """k控制每个party拥有几类label"""
    #各个样本旗下的特征序列
    party_dataset=[[] for _ in range(args.num_parties)]
    #各个样本旗下的标签序列
    party_label=[[] for _ in range(args.num_parties)]
    #获取各个label下的样本特征序列
    label_example_features,num_labels,num_features = get_label_features(args,tokenizer)
    #各个标签下有哪些party
    label_contain_party=[[] for _ in range(num_labels)]
    for i in range(args.num_parties):
        party_label[i]=random.sample(range(0,num_labels),args.k)

        for j in range(num_labels):
            #如果j标签在第i个party的标签列表中
            if j in party_label[i]:
                #则将第i个party加到第j个label的party序列中
                label_contain_party[j].append(i)

    for i in range(num_labels):
        # 统计第i个label中共有几个party
        label_contain_party_num = len(label_contain_party[i])

        # 第i个label的样本序列长度
        current_label_example_features = np.array(label_example_features[i])
        current_label_example_features_len = len(current_label_example_features)
        # 生成索引序列
        idx = np.random.permutation(current_label_example_features_len)
        if label_contain_party_num > 0:
            # 将该label划分给旗下的各party的索引序列
            batch_idx = np.array_split(idx, label_contain_party_num)
            for index, j in enumerate(label_contain_party[i]):
                party_dataset[j].extend(current_label_example_features[batch_idx[index]])
    return party_dataset


"""
4.数量分布偏移(尽量避免标签分布偏移的情况存在)
"""
def non_iid_quantity(args,tokenizer):
  party_dataset = [[] for _ in range(args.num_parties)]
  # 各个label旗下的样本特征序列
  label_example_features,num_labels,num_features = get_label_features(args,tokenizer)
  #确定狄利克雷分布
  proportions=np.random.dirichlet(np.repeat(args.beta,args.num_parties))
  proportions=proportions/proportions.sum()
  # 将各个label旗下的样本根据狄利克雷的比例分配给每个party
  for i in range(num_labels):
      # 该label旗下的样本特征序列
      current_label_example_features = np.array(label_example_features[i])
      # 获取该label下的样本数量
      current_label_example_features_len = len(current_label_example_features)
      # 随机打乱标签索引
      idxs = np.random.permutation(current_label_example_features_len)
      #确定分配比例
      pro= (np.cumsum(proportions) * current_label_example_features_len).astype(int)[:-1]
      # 根据客户端数量划分该label的索引
      batch_idxs = np.split(idxs, pro)
      for j in range(args.num_parties):
          # 不将被追加的列表或元组当成一个整体，而是只追加列表中的元素，则可使用列表提供的 extend() 方法
          party_dataset[j].extend(current_label_example_features[batch_idxs[j]])
  return party_dataset
# def non_iid_quantity(args,tokenizer):

#     # 各个label旗下的样本特征序列
#     label_example_features,num_labels,num_features = get_label_features(args,tokenizer)
#     min_length=0
#     while min_length<=0:
#         party_dataset = [[] for _ in range(args.num_parties)]
#         #确定狄利克雷分布
#         proportions=np.random.dirichlet(np.repeat(args.beta,args.num_parties))
#         proportions=proportions/proportions.sum()
#         # 将各个label旗下的样本根据狄利克雷的比例分配给每个party
#         for i in range(num_labels):
#             # 该label旗下的样本特征序列
#             current_label_example_features = np.array(label_example_features[i])
#             # 获取该label下的样本数量
#             current_label_example_features_len = len(current_label_example_features)
#             # 随机打乱标签索引
#             idxs = np.random.permutation(current_label_example_features_len)
#             #确定分配比例
#             pro= (np.cumsum(proportions) * current_label_example_features_len).astype(int)[:-1]
#             # 根据客户端数量划分该label的索引
#             batch_idxs = np.split(idxs, pro)
#             for j in range(args.num_parties):
#                 # 不将被追加的列表或元组当成一个整体，而是只追加列表中的元素，则可使用列表提供的 extend() 方法
#                 party_dataset[j].extend(current_label_example_features[batch_idxs[j]])
#         min_length=min(len(party_dataset[j]) for j in range(args.num_parties))
#     return party_dataset
# party_dataset=non_iid_quantity(num_parties,num_label,data_dir,task,tokenizer,max_seq_length,local_rank,beta=100)
# for i in range(num_parties):
#     print(f'party{i}:{len(party_dataset[i])}')


# def non_iid_quantity(args,tokenizer):
#     min_length=0
#     features, num_lables, num_features = get_all_label_features(args, tokenizer)
#     while min_length<=10:
#         #确定狄利克雷分布
#         proportions=np.random.dirichlet(np.repeat(args.beta,args.num_parties))
#         proportions=proportions/proportions.sum()
#         min_length=min(proportions*num_features)
#     proportions = (np.cumsum(proportions)*num_features).astype(int)[:-1]
#     # 确定样本总数
#     num_global_train_data = num_features
#     # 2.生成全局训练样本的索引并随机打乱
#     idxs = np.random.permutation(num_global_train_data)
#     # 3.根据客户端数划分全局训练样本索引
#     batch_idxs = np.array_split(num_global_train_data,proportions)
#     # 4.根据batch索引取出全局训练样本中对应的样本
#     party_dataset = [[] for _ in range(args.num_parties)]
#     for i in range(args.num_parties):
#         party_dataset[i].extend(np.array(features)[batch_idxs[i]])       
#     return party_dataset

"""
5.标签分布偏移和数量分布偏移同时存在
在标签分布偏移的基础上，控制数量分布偏移的程度
"""
def non_iid_label_and_quantity(args,tokenizer):
  party_dataset=[[]for _ in range(args.num_parties)]
  """
  :param beta: beta用于控制标签分布偏移的程度,beta越大，偏移程度越小
  :param gamma: alpha用于控制数量分布偏移的程度，alpha越大，偏移程度越大
  """
  # 各个label旗下的样本特征序列
  label_example_features,num_labels,num_features = get_label_features(args,tokenizer)
  min_length,max_length=0,3000
  a,b=num_features/args.num_parties-args.gamma,num_features/args.num_parties+args.gamma

  while min_length<a or max_length>b:
      party_dataset=[[]for _ in range(args.num_parties)]
      all_batch_idxs=[[] for _ in range(args.num_parties)]
      #将各个label旗下的样本按照狄利克雷分布分配给每个party
      for i in range(num_labels):
          #该label旗下的样本特征序列
          current_label_example_features=np.array(label_example_features[i])
          #获取该label下的样本数量
          current_label_example_features_len=len(current_label_example_features)
          #生成样本索引
          idxs=np.random.permutation(current_label_example_features_len)
          #生成狄利克雷分布
          proportions=np.random.dirichlet(np.repeat(args.beta,args.num_parties))
          proportions=proportions/proportions.sum()
          proportions = (np.cumsum(proportions) * current_label_example_features_len).astype(int)[:-1]
          #根据狄利克雷分布划分该label的索引
          batch_idxs=np.split(idxs,proportions)
          # print(f'label{i}:{batch_idxs}')
          # print('*'*20)
          for j in range(args.num_parties):
              #不将被追加的列表或元组当成一个整体，而是只追加列表中的元素，则可使用列表提供的 extend() 方法
              party_dataset[j].extend(current_label_example_features[batch_idxs[j]])
          for j in range(args.num_parties):
              all_batch_idxs[j].extend(batch_idxs[j])
      min_length=min(len(party_label) for party_label in all_batch_idxs)
      max_length=max(len(party_label) for party_label in all_batch_idxs)
  # print(min_length)
  # print(max_length)
  return party_dataset
#
# party_dataset=non_iid_label_and_quantity(num_parties,num_label,data_dir,task,tokenizer,
#                                     max_seq_length,local_rank,num_features=2000,beta=1,alpha=100,shreshold=100)
# for i in range(num_parties):
#     print(f'party{i}:{len(party_dataset[i])}')

"""
生成数据集
"""
def load_dataset(party_i_dataset):
    """
    将上述party_dataset中第i个客户端的样本特征序列转换为数据集
    """
    all_input_ids = torch.tensor([f.input_ids for f in party_i_dataset], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in party_i_dataset], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in party_i_dataset], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in party_i_dataset], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in party_i_dataset], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset

# party_0_dataset=load_dataset(party_dataset[0])
# # print(len(party_0_dataset))
# sampler=RandomSampler(party_0_dataset)
# party_0_dataloader=DataLoader(party_0_dataset,batch_size=8,sampler=sampler,collate_fn=collate_fn)
# # print(len(party_0_dataloader))


#1.iid quantity
# party_dataset=iid_quantity(num_parties,data_dir,task,tokenizer,max_seq_length,local_rank)
# party_0_dataset=load_dataset(party_dataset[0])
# print(len(party_0_dataset))
# sampler=RandomSampler(party_0_dataset)
# party_0_dataloader=DataLoader(party_0_dataset,batch_size=8,sampler=sampler,collate_fn=collate_fn)


partition={
    'iid_quantity':iid_quantity,
    'iid_label':iid_label,
    'non_iid_quantity':non_iid_quantity,
    'non_iid_label':non_iid_label,
    'non_iid_label_k':non_iid_label_k,
    'non_iid_label_quantity_coexist':non_iid_label_and_quantity
}