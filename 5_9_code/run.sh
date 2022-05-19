python myexperiment_420_3.py\
  --model_name_or_path='pre_trained_model/albert_small_zh'\
  --task_name='online'\
  --data_dir='dataset/online shopping'\
  --fed_output_dir='fed_output'\
  --vocab_file='pre_trained_model/albert_small_zh/vocab.txt'\
  --model_type='albert_small_zh'\
  --classifier_lr=1e-3\
  --fed_num_train_epochs=10\
  --fed_train_batch_size=32\
  --fed_eval_batch_size=32\
  --algorithm='fedagg_prox'\
  --num_parties=100\
  --sample=0.03\
  --communication_rounds=100\
  --partition_strategy='iid_label'\
  --beta=1\
  --alpha=1\
  --acc_rounds=5\
  --mu=1\
  --save_rounds=100\
  --k=3\






