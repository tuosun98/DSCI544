# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import sys
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score,precision_score,recall_score,roc_auc_score
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def eval_criterias(preds, labels,pred_pro):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    prec = precision_score(y_true=labels,y_pred=preds,average='macro')
    recall = recall_score(y_true=labels,y_pred=preds,average='macro')
    auc = roc_auc_score(labels, pred_pro)

    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "prec":prec,
        "recall":recall,
        "auc":auc,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels,pred_pro):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return eval_criterias(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return eval_criterias(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "lcqmc":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "tnews":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "xnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "inews":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "my":
        return pearson_and_spearman(pred_pro,labels)
    elif task_name == "return":
        return pearson_and_spearman(pred_pro,labels)
    else:
        raise KeyError(task_name)
