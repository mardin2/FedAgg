import numpy as np
import matplotlib
import matplotlib.pyplot as plt


"""
热力图，每个label分配给每个party的样本数有多少
"""
from processors.dataset_partition import get_label_features

def non_iid_label_heatmap(args,tokenizer):
    # seed_everything(args.seed)
    label_example_features,num_labels,num_features=get_label_features(args,tokenizer)
    label_party_quantity=[[] for _ in range(num_labels)]
    for i in range(num_labels):
        #该label旗下的样本特征序列
        current_label_example_features=np.array(label_example_features[i])
        #获取该label下的样本数量
        current_label_example_features_len=len(current_label_example_features)
        #随机打乱标签索引
        idxs=np.random.permutation(current_label_example_features_len)
        #根据客户端数量划分该label的索引
        batch_idxs=np.array_split(idxs,args.num_parties)
        label_party_quantity[i].append(len(batch_idxs[j]) for j in range(args.num_parties))

    labels=np.arange(num_labels)
    parties=np.arange(args.num_parties)
    fig,ax=plt.subplots()
    im=ax.imshow(label_party_quantity)
    ax.set_xticks(np.arange(len(args.num_parties)))
    ax.set_yticks(np.arange(len(num_labels)))
    ax.set_xticklabels(parties)
    ax.set_yticklabels(labels)
    for i in range(num_labels):
        for j in range(args.num_parties):
            text=ax.text(j,i,label_party_quantity[i,j],ha='center',va='center',color='w')
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()




