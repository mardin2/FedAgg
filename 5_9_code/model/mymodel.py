from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import os
import sys
import torch
from torch import nn
from torch.nn.modules import LayerNorm as AlbertLayerNorm
from torch.nn import CrossEntropyLoss, MSELoss

from model.modeling_albert import ACT2FN, AlbertPreTrainedModel, AlbertPooler
from model.modeling_utils import PreTrainedModel, prune_linear_layer
from model.configuration_albert import AlbertConfig
from model.file_utils import add_start_docstrings
from model.tokenization_bert import BertTokenizer

"""1.检查albert的embedding层（三个embedding到embedding_size后，再映射到hidden_size(在AlbertModel中需要改)"""
class AlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        self.LayerNorm = AlbertLayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # embeddings=self.embedding_hidden_mapping_in(embeddings)
        return embeddings

tokenizer=BertTokenizer.from_pretrained('../prev_trained_model/albert_small_zh')
config=AlbertConfig.from_pretrained('../prev_trained_model/albert_small_zh')
model=AlbertEmbeddings(config)
batch_size,sequence_length=20,20
input_id=torch.arange(400).reshape((20,20))
input_id_embedding=model(input_id)
# print(input_id_embedding.shape)#torch.Size([20, 20, 384])

"""2.检查self-attention层（仅聚合多头，不做后续处理）"""
class AlbertSelfAttention(nn.Module):
    """
    自注意力层，不包括对多头合并的注意力进行transformer的过程
    输入：上一个encoder或者embedding层的输出
    输出：(多头合并context_layer：batch_size,sequence_length,hidden_size,attention_probs:batch_size,num_heads,sequence_length,attention_head_size)
    """
    def __init__(self, config):
        super(AlbertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        """output_attentions:是否要输出注意力得分"""
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """
        :param x: batch_size,sequence_length,hidden_size
        :return: batch_size,num_attention_heads,sequence_length,attention_head_size
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # extended_attention_mask = extended_attention_mask.to(
            #     dtype=next(self.parameters()).dtype)  # fp16 compatibility
            # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            """
            attention_mask.shape:batch_size,1,1,sequence_length(在BertModel的forward()函数中被重写了
            对于真实value：attention_mask=0
            对于pad：attention_mask=-10000(softmax后接近0)
            """
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        """context_layer:多个头合并后得到：batch_size,sequence_length,hidden_size"""
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


# model=AlbertSelfAttention(config)
# attention_mask=torch.ones_like(input_id)
# self_attention_output=model(hidden_states=input_id_embedding,attention_mask=attention_mask)
# print(f'context_layer:{self_attention_output[0].shape}')
# print(f'attention_probs:{self_attention_output[1].shape}')

"""检查self_attention对合并多头的注意力的transformer"""
class AlbertSelfOutput(nn.Module):
    """
    self_attention后的将多个头合并得到的context经过transformer的操作
    """
    def __init__(self, config):
        super(AlbertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# model=AlbertSelfOutput(config)
# attention_linear_output=model(self_attention_output[0])
# print(f'self_attention_shape:{attention_linear_output.shape}')#torch.Size([20, 20, 384])

"""3.检查self-attention和其后的linear projection合起来的attention的输出"""
class AlbertAttention(nn.Module):
    """
    输入：前一个encoder层的输出或embedding层的输出
    输出：(linear projection后的self-attention,仅仅合并的多头context,attention_probs)
    """
    def __init__(self, config):
        super(AlbertAttention, self).__init__()
        self.self = AlbertSelfAttention(config)
        self.output = AlbertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,self_outputs)
        """
        self.outputs:(1)self_attention后的多头合并得到的context,(2)attention_prob
        attention_output:transformer后的self-attention
        """
        return outputs

# model=AlbertAttention(config)
# attention_output=model(input_tensor=input_id_embedding,attention_mask=attention_mask)
# print(f'attention_output:{attention_output[0].shape}')
# print(f'context_layer:{attention_output[1][0].shape}')
# print(f'attention_probs:{attention_output[1][1].shape}')


class AlbertOutput(nn.Module):
    def __init__(self, config):
        super(AlbertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class AlbertIntermediate(nn.Module):
    def __init__(self, config):
        super(AlbertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = AlbertOutput(config)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        intermediate_output = self.dense(hidden_states)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        output = self.output(intermediate_output)
        return output

"""5.检查ffn层"""
class AlbertFFN(nn.Module):
    """
    输入：self_attention的linear_projection后的context
    输出：经过两个全连接层的输入
    """
    def __init__(self, config):
        super(AlbertFFN, self).__init__()
        self.intermediate = AlbertIntermediate(config)

    def forward(self, attention_output):
        output = self.intermediate(attention_output)
        return output

# model=AlbertFFN(config)
# ffn_output=model(attention_output[0])
# print(ffn_output.shape)#torch.Size([20, 20, 384])

"""6.检查一个albert layer"""
class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()
        self.attention = AlbertAttention(config)
        self.ffn = AlbertFFN(config)
        self.LayerNorm_1 = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm_2 = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self.LayerNorm_1(attention_outputs[0] + hidden_states)
        ffn_output = self.ffn(attention_output)
        ffn_output = self.LayerNorm_2(ffn_output+attention_output)
        outputs = (ffn_output,) + attention_outputs[1:] # add attentions if we output them
        return outputs

# model=AlbertLayer(config)
# layer_output=model(input_id_embedding,attention_mask=attention_mask)
# print(f'ffn_output:{layer_output[0].shape}')
# print(f'context_layer:{layer_output[1][0].shape}')
# print(f'attention_probs:{layer_output[1][1].shape}')

"""6.一个layer中attention和ffn重复的次数"""
class AlbertGroup(nn.Module):
    def __init__(self, config):
        super(AlbertGroup, self).__init__()
        """
        inner_group_num:The number of inner repetition of attention and ffn
        """
        self.inner_group_num = config.inner_group_num
        self.inner_group = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_attentions = ()
        layer_hidden_states = ()
        for inner_group_idx in range(self.inner_group_num):
            """该layer中的第i个模块"""
            layer_module = self.inner_group[inner_group_idx]
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask)
            hidden_states = layer_outputs[0]
            layer_attentions = layer_attentions + (layer_outputs[1],)
            layer_hidden_states = layer_hidden_states + (hidden_states,)
        return (layer_hidden_states, layer_attentions)
# model=AlbertGroup(config)
# group_output=model(input_id_embedding,attention_mask=attention_mask)
# print(f'layer_hidden_states:{group_output[0][0].shape}')
# print(f'layer_attention:{(group_output[1][0][0]).shape}')

class AlbertTransformer(nn.Module):
    def __init__(self, config):
        super(AlbertTransformer, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.num_hidden_layers = config.num_hidden_layers
        self.num_hidden_groups = config.num_hidden_groups
        """
        num_hidden_groups:同一group的layer共享相同的参数，num_hidden_groups控制共有几个group
        """
        self.group = nn.ModuleList([AlbertGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for layer_idx in range(self.num_hidden_layers):
            if self.output_hidden_states and layer_idx == 0:
                """？？？？加上embedding层的hidden_states"""
                all_hidden_states = all_hidden_states + (hidden_states,)
            """
            layer_idx/num_hidden_layers:[0,1)   *num_hidden_groups:[0,num_hidden_groups)
            所以可以将num_hidden_layers个layers划分进num_hidden_groups个group里
            每个group中的layer因为group_idx相同，因此共用一个layer_module
            """
            group_idx = int(layer_idx / self.num_hidden_layers * self.num_hidden_groups)
            layer_module = self.group[group_idx]
            layer_outputs = layer_module(hidden_states, attention_mask)
            """将该layer_module中的最后一个ffn的输出取出"""
            hidden_states = layer_outputs[0][-1]
            if self.output_attentions:
                """将该layer_module中的所有module中的self-attention的context和attention_probs"""
                all_attentions = all_attentions + layer_outputs[1]
            if self.output_hidden_states:
                """将该layer_module中的所有module中的ffn的输出"""
                all_hidden_states = all_hidden_states + layer_outputs[0]
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
# model=AlbertTransformer(config)
# transformer_output=model(input_id_embedding,attention_mask=attention_mask)
# print(f'last_layer_ffn_output:{transformer_output[0].shape}')#last_layer_ffn_output:torch.Size([20, 20, 384])
# print(f'emb_hiddens_state+{config.num_hidden_groups}hidden_group_states.len:{len(transformer_output[1])}')#7
# print(f'第一个hidden_gropu_states:{transformer_output[1][1].shape}')#第一个hidden_gropu_states:torch.Size([20, 20, 384])
# print(transformer_output[2][0][1].shape)#torch.Size([20, 12, 20, 20])

"""
将num_hidden_groups个hidden_states沿着hidden_size的维度并起来
"""
# concat_hidden_states=torch.cat(transformer_output[1][1:],dim=-1)
# print(f'concat_hidden_states:{concat_hidden_states.shape}')

class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super(AlbertEncoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = config.embedding_size
        self.embedding_hidden_mapping_in = nn.Linear(self.embedding_size, self.hidden_size)
        self.transformer = AlbertTransformer(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        if self.embedding_size != self.hidden_size:
            prev_output = self.embedding_hidden_mapping_in(hidden_states)
        else:
            prev_output = hidden_states
        outputs = self.transformer(prev_output, attention_mask, head_mask)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

model=AlbertEncoder(config)
attention_mask=torch.ones((batch_size,sequence_length))
# encoder_output=model(input_id_embedding,attention_mask=attention_mask)
# concat_hidden_states=torch.cat(encoder_output[1][1:],dim=-1)
# print(f'concat_hidden_states:{concat_hidden_states.shape}')
# print(encoder_output[0].shape)


class AlbertModel(AlbertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """



    def __init__(self, config):
        super(AlbertModel, self).__init__(config)

        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        # self.pooler=AlbertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        #sequence_output:最后一个ffn的输出
        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, (hidden_states), (attentions)

model=AlbertModel(config)
model_output=model(input_id,attention_mask)
print(f'sequence_output:{model_output[0].shape}')#sequence_output:torch.Size([20, 20, 384])
print(f'num of hidden_state:{len(model_output[1])}')#num of hidden_state:7
print(f'第一个group的一个module的hidden_state:{model_output[1][1].shape}')#第一个group的一个module的hidden_state:torch.Size([20, 20, 384])
concat_hidden_states=torch.cat(model_output[1][1:],dim=-1)
concat_cls_hidden_states=concat_hidden_states[:,0,:]
print(f'concat_hidden_states:{concat_hidden_states.shape}')#concat_hidden_states:torch.Size([20, 20, 2304])
print(concat_cls_hidden_states.shape)
print('='*20)

class AlbertMyClassifier(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertMyClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        # print(config.num_labels)

        self.albert = AlbertModel(config)
        # self.dropout = nn.Dropout(0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        if config.inner_group_num==1:
            num_inputs=config.hidden_size*int(config.num_hidden_layers/config.num_hidden_groups)

        self.classifier = nn.Sequential(nn.Linear(num_inputs, config.hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(
                                            0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob),
                                        nn.Linear(config.hidden_size, self.config.num_labels))

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.albert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask)

        # 将所有层的隐藏层状态拼在一起
        concat_hidden_states = torch.cat(outputs[1][1:], dim=-1)
        # 取出cls对应的所有隐藏层状态
        concat_hidden_states = concat_hidden_states[:, 0]
        # 将cls对应的拼接隐藏层状态放进分类器中
        concat_output = self.classifier(concat_hidden_states)

        outputs = (concat_output,) + outputs[1:]  # add hidden states and attention if they are here

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(concat_output.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
# model=AlbertMyClassifier(config)
# labels=torch.ones(batch_size,dtype=torch.long)
# classifier_output=model(input_id,attention_mask=attention_mask,labels=labels)
# print(f'loss:{classifier_output[0]}')#loss:0.7980676889419556
# print(f'logits:{classifier_output[1].shape}')#logits:torch.Size([20, 2])
# print(len(classifier_output[2]))#7
no_decay = ['bias', 'LayerNorm.weight']
classifier_params = ['classifier']
# for n,p in model.named_parameters():
#
#     if not any(nd in n for nd in classifier_params) and any(nd in n for nd in no_decay):
#         print(n)

#1.classifier层的weight：参与衰减，学习率=classifier的学习率
# for n,p in model.classifier.named_parameters():
#     if not any(nd in n for nd in no_decay):
#         print(n)
#         '''
#         0.weight
#         3.weight
#         '''

#2.classifier层bias:不参与权重衰减，学习率：classifier的学习率
# for n,p in model.classifier.named_parameters():
#     if any(nd in n for nd in no_decay):
#         print(n)
#         """
#         0.bias
#         3.bias
#         """

#3.其它层的bias和layernormweight：不参与权重衰减，学习率：默认学习率
# for n,p in model.named_parameters():
#     if not any(nd in n for nd in classifier_params) and not any(nd in n for nd in no_decay):
#         print(n)
"""
albert.embeddings.word_embeddings.weight
albert.embeddings.position_embeddings.weight
albert.embeddings.token_type_embeddings.weight
albert.encoder.embedding_hidden_mapping_in.weight
albert.encoder.transformer.group.0.inner_group.0.attention.self.query.weight
albert.encoder.transformer.group.0.inner_group.0.attention.self.key.weight
albert.encoder.transformer.group.0.inner_group.0.attention.self.value.weight
albert.encoder.transformer.group.0.inner_group.0.attention.output.dense.weight
albert.encoder.transformer.group.0.inner_group.0.ffn.intermediate.dense.weight
albert.encoder.transformer.group.0.inner_group.0.ffn.intermediate.output.dense.weight
albert.encoder.transformer.group.0.inner_group.0.LayerNorm_1.weight
albert.encoder.transformer.group.0.inner_group.0.LayerNorm_2.weight
"""

#4.其它层的weight：参与权重衰减，学习率：默认学习率
# for n,p in model.named_parameters():
#     if not any(nd in n for nd in classifier_params) and any(nd in n for nd in no_decay):
#         print(n)
"""
albert.embeddings.LayerNorm.weight
albert.embeddings.LayerNorm.bias
albert.encoder.embedding_hidden_mapping_in.bias
albert.encoder.transformer.group.0.inner_group.0.attention.self.query.bias
albert.encoder.transformer.group.0.inner_group.0.attention.self.key.bias
albert.encoder.transformer.group.0.inner_group.0.attention.self.value.bias
albert.encoder.transformer.group.0.inner_group.0.attention.output.dense.bias
albert.encoder.transformer.group.0.inner_group.0.ffn.intermediate.dense.bias
albert.encoder.transformer.group.0.inner_group.0.ffn.intermediate.output.dense.bias
albert.encoder.transformer.group.0.inner_group.0.LayerNorm_1.bias
albert.encoder.transformer.group.0.inner_group.0.LayerNorm_2.bias
"""

class AlbertPreClassifier(AlbertPreTrainedModel):
    """
    原始的方法：即拿出最后一层的cls的隐藏层状态做分类
    输出：tuple中四个元素
    1.outputs[0]:loss
    2.output[1]:logit预测:batch_size*num_labels
    3.all_hidden_states
    4.all_attention_probs
    """
    def __init__(self, config):
        super(AlbertPreClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        # print(config.num_labels)

        self.albert = AlbertModel(config)
        # self.dropout = nn.Dropout(0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob)

        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(
                                            0.1 if config.hidden_dropout_prob == 0 else config.hidden_dropout_prob),
                                        nn.Linear(config.hidden_size, self.config.num_labels))

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.albert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask)

        # 取出cls对应的所有隐藏层状态
        last_hidden_states =outputs[0]
        cls_hidden_state=last_hidden_states[:,0]
        # 将cls对应的拼接隐藏层状态放进分类器中
        concat_output = self.classifier(cls_hidden_state)

        outputs = (concat_output,) + outputs[1:]  # add hidden states and attention if they are here

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(concat_output.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
no_decay = ['bias', 'LayerNorm.weight']
classifier_params = ['classifier']
model=AlbertPreClassifier(config)
for n,p in model.named_parameters():
    print(n)
