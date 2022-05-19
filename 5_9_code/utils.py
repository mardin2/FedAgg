import torch
import numpy as np
from metrics.glue_compute_metrics import simple_accuracy
from sklearn.metrics import f1_score
from pathlib import Path
import logging

def compute_metrics(model,data_loader,device):
    preds = None
    label = None
    model.to(device)
    model.eval()
    for step, batch in enumerate(data_loader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
        if preds is None:
            preds = logits.detach().cpu().numpy()
            label = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            label = np.append(label, inputs['labels'].detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    accuracy=simple_accuracy(preds,label)
    f1 = f1_score(y_true=label, y_pred=preds, average='micro')
    return accuracy,f1



logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file,Path):
        log_file = str(log_file)

    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger