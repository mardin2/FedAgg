""" GLUE processors and helpers """

import logging
import os
import torch
import pandas as pd
import numpy as np
from processors.utils import DataProcessor, InputExample, InputFeatures
from model.tokenization_bert import BertTokenizer

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    # print(f'zip(*batch):{zip(*batch)}')
    # for a in zip(*batch):
    #     print(a)
    #     print('*'*20)
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def glue_convert_examples_to_features(examples, tokenizer,
                                      max_seq_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    # print(label_map)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b  =None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            token_type_ids.append(0)
        tokens.append("[SEP]")
        token_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                token_type_ids.append(1)
            tokens.append("[SEP]")
            token_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        if output_mode == "classification":
            label_id = label_map[example.label]
            # print(label_id)
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("input length: %d" % (input_len))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label_id,
                          input_len=input_len))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()



class TnewsProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """得到训练样本"""
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "toutiao_category_train_big.txt")), "train")[50000:150000]

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "toutiao_category_dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "toutiao_category_test_supplement.txt")), "test")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(17):
            if i == 5 or i == 11:
                continue
            labels.append(str(100 + i))
        # print(labels)
        return labels

    #得到映射过的样本标签(用在：fed的数据划分中)
    def get_label_map(self):
        label_list=TnewsProcessor().get_labels()
        label_list_map=[]
        for i,label in enumerate(label_list):
            label_list_map.append(i)
        return label_list_map



    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """
        set_type:train/dev/test
        guid:数据集中的标识,train-1,dev-1,test-1
        label:样本的标签
        """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            # if set_type == 'test':
            #     label = '0'
            # else:
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class OnlineProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """得到训练样本"""
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "online_shopping_train.csv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "online_shopping_test.csv")), "test")

    def get_labels(self):
        """See base class."""
        labels =[str(0),str(1)]
        return labels
    
    #得到映射过的样本标签(用在：fed的数据划分中)
    def get_label_map(self):
        label_list=OnlineProcessor().get_labels()
        label_list_map=[]
        for i,label in enumerate(label_list):
            label_list_map.append(i)
        return label_list_map

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        """
        set_type:train/dev/test
        guid:数据集中的标识,train-1,dev-1,test-1
        label:样本的标签
        """
        examples = []
        for i in range(len(lines)):
            label=str(lines[i][0])
            text_a=lines[i][1]
            guid="%s-%s" % (set_type, i)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



glue_tasks_num_labels = {
    'tnews': 15,
    'online': 2
}

"""字典,key:task名称，值：task对应的预处理方法processor"""
glue_processors = {
    'tnews': TnewsProcessor,
    'online': OnlineProcessor
}

"""输出任务是：分类/回归"""
glue_output_modes = {
    'tnews': "classification",
    'online': 'classification'
}

# tnews_processor=TnewsProcessor()
# train_examples=tnews_processor.get_train_examples('../dataset/tnews/')
# tokenizers=BertTokenizer.from_pretrained('../prev_trained_model/albert_small_zh')
# label_list=tnews_processor.get_labels()
# # print(type(label_list))
# label_map = {label: i for i, label in enumerate(label_list)}
# # print(label_map['101'])
# train_features=glue_convert_examples_to_features(train_examples,tokenizers,task='tnews')
# # print(train_features)
# for i in range(5):
#     print(f'example{i},input_ids:{train_features[i].input_ids}')
#     print(f'example{i},attention_mask:{train_features[i].attention_mask}')
#     print(f'example{i},token_type_id:{train_features[i].token_type_ids}')
#     print(f'example{i},label:{train_features[i].label}')
#     print(f'example{i},input_len:{train_features[i].input_len}')



