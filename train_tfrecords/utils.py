"""
Contains methods for 
    - preprocessing the datasets (data_generator.py) and 
    - calculating the performance metrics (training.py) used to evaluate the model(s).
"""

import logging
import csv
import os
import numpy as np

from datetime import datetime, timedelta
from glob import glob
from sklearn.metrics import confusion_matrix


def create_logger():
    """Setup the logging environment"""
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    format_str = '%(asctime)s [%(process)d] [%(levelname)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)


def parse_date(start_d):
    assert isinstance(start_d, str), "start date must be in form 'now' or 'y.m.d.h.m'"
    if start_d == 'now':
        return datetime.now()
    else:
        return datetime(*map(lambda t: int(t), start_d.split('.')))
    
    
def find_training_data(start_date,train_period,train_data_path):
    """Locate the data files for training.
    Args:
        start_date (str): Current date
        train_period (int): Number of previous days to use for training
    Returns:
        data_file (list of str): List containing location of H5files
    """

    TF_file = []
    now_time = parse_date(start_date)
    delta_time = timedelta(train_period)

    for f_name in os.listdir(train_data_path):
        #if ('h5' in f_name or 'tfrecords' in f_name) and 'patch' not in f_name:
        #    f_time_str = f_name.replace('train_data', '').replace('.h5', '').replace('.tfrecords', '')
        try:
            if ('h5' in f_name or 'tf' in f_name) and 'patch' not in f_name:
                f_time_str = f_name.replace('train_data_', '').replace('.h5', '').replace('.tf', '')        
                f_time = datetime.strptime(f_time_str, '%Y-%m-%dT%H:%M:%S.%f')
                if now_time > f_time > now_time + delta_time:
                    TF_file.append(train_data_path+f_name)
        except:
            logging.info('{0} file name not matching the format'.format(f_name))
    return TF_file


def find_latest_model_dir(save_dir,store_dir, name='MLR'):
    """Parse model folder and find latest model parsed by time
    Args:
        save_dir: path to save models
        store_dir: path to store models
        name: name of model ,default as 'MLR'
    Returns:
        str: model folder directory
    """
    path = save_dir + name + '_'
    directories = [i for i in glob(path + '*')]
    dir_date_list = [i.split(sep=path)[1] for i in directories]
    if len(dir_date_list)>=1:
        model = name + '_' + max(dir_date_list)
    else:
        model = store_dir
    logging.info('model: %s_%s',name,model)
    return model


def output2csv(data_result, Colnames, dir_hps):
    """Stores the training result into a csv file."""    
    if not os.path.isfile(dir_hps.result_dir):
        with open(dir_hps.result_dir, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['Model_name', 'training_time'] + Colnames)
            
    with open(dir_hps.result_dir, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data_result)


def safe_division(x, y):
    try:
        return x / (y + 1e-15)
    except (ZeroDivisionError, RuntimeWarning):
        return 0


def table_metrics(t_y, predicted_class):
    """Compute the evaluation metrics for the 2-class classification model.
    The classes are too imbalanced, evaluate using precision and recall.

    For reference of computation:
        * Sensitivity or Recall or TPR : TP / (TP+FN)
        * FPR                          : FP / (FP + TN)
        * Specificity or TNR           : TN / (TN+FP)
        * Precision or PPV             : TP / (TP+FP)
        * F1-score                     : 2TP / (2TP + FP + FN)

    Positive class: 1

    :param t_y: ground truth class
    :param predicted_class: predicted probability
    :return: accuracy, recall, precision, f1_score
    """
    try:
        cm = confusion_matrix(t_y, predicted_class)
        _, _, FP, FN, TP = sum(sum(cm)), cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        #accuracy = safe_division(TP + TN, total)
        recall = safe_division(TP, TP + FN)
        precision = safe_division(TP, TP + FP)
        f1_score = safe_division(2 * TP, (2 * TP) + FP + FN)

        return recall, precision, f1_score
    except Exception as err:
        logging.info('[Error] Confusion matrix: %s', str(err))


def IU_method(y, y_hat, cut):
    import numpy as np

    y = y.flatten()
    y_hat = y_hat.flatten()
    
    sen = sum(y[[i > cut for i in y_hat]]) / len(y)
    spc = np.count_nonzero(y[[i < cut for i in y_hat]] == 0) / len(y)
    try:
        auc = roc_auc_score(y, y_hat)
        return -1 * (abs(sen - auc) + abs(spc - auc))  # set to negative for maximization
    except:
        # auc = 0.5
        return None


def ER_method(y, y_hat, cut):
    sen = sum(y[[i > cut for i in y_hat]]) / len(y)
    spc = np.count_nonzero(y[[i < cut for i in y_hat]] == 0) / len(y)
    return -1 * np.sqrt((1 - sen) ** 2 + (1 - spc) ** 2)


def CZ_method(y, y_hat, cut):
    sen = sum(y[[i > cut for i in y_hat]]) / len(y)
    spc = np.count_nonzero(y[[i < cut for i in y_hat]] == 0) / len(y)
    return sen * spc


def YD_method(y, y_hat, cut):
    sen = sum(y[[i > cut for i in y_hat]]) / len(y)
    spc = np.count_nonzero(y[[i < cut for i in y_hat]] == 0) / len(y)
    return sen + spc - 1



def get_cutpoint_method(y, y_hat, _cut_list, method):
    """ Calculate cut index for AUC given a list of cut point

    :param method: method for AUC cut point method 'IU', 'EU', 'CZ','YD'
    :param y: ground truth class
    :param y_hat: predicted probability
    :param _cut_list: list of index to be cut
    :return: list of index value for certain cut point in _cut_list
    """
    if method == 'IU':
        return [IU_method(y, y_hat, cut) for cut in _cut_list]
    elif method == 'ER':
        return [ER_method(y, y_hat, cut) for cut in _cut_list]
    elif method == 'CZ':
        return [CZ_method(y, y_hat, cut) for cut in _cut_list]
    elif method == 'YD':
        return [YD_method(y, y_hat, cut) for cut in _cut_list]



def calculate_confusion_cutpoint(y, y_hat, method='IU', num_of_cut=False):
    """
    Calculate cut index for AUC cut point
    :param y: ground truth class
    :param y_hat: predicted probability
    :param method: method for AUC cut point method 'IU', 'EU', 'CZ','YD'
    :param num_of_cut: number of bins in AUC calculation
    :return: cut_list, index_list, cut point
    """
    if not num_of_cut:
        num_of_cut = np.float(len(y) - 1)
    cut_list = np.arange(0.0, 1.0, 1 / num_of_cut)
    index_list = get_cutpoint_method(y, y_hat, cut_list, method)
    try:
        cut_point = cut_list[np.where(index_list == np.max(index_list))[0][0]]
    except:
        cut_point = None
    return cut_list, index_list, cut_point


def plot_metrics(y, y_hat, data, dir_hps, pos_label=1):
    """Use for visualization the quality of results.

    Return:
        * Histogram of CTR prediction
        * ROC curve
        * Precision-Recall curve
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
    from matplotlib import pyplot as plt
    from sklearn.utils.fixes import signature

    plt.figure(figsize=(5, 4))
    plt.hist(y_hat, 20)
    plt.xlabel('CTR Prediction')
    plt.ylabel('Frequency')
    plt.xlim(0., 1.)
    plt.title(data + ': predicted CTR')
    plt.savefig(dir_hps.save_dir + dir_hps.store_dir + '/predictedCTR' + data + '.pdf', bbox_inches="tight")

    auc = roc_auc_score(y, y_hat) * 100
    fpr, tpr, _ = roc_curve(y, y_hat, pos_label=pos_label)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(data + ' ROC curve:\nAUC={0:0.2f}'.format(auc))
    plt.xlabel('False positive rate')
    plt.savefig(dir_hps.save_dir + dir_hps.store_dir + '/rocCurve' + data + '.pdf', bbox_inches="tight")

    precision, recall, _ = precision_recall_curve(y, y_hat, pos_label=pos_label)
    average_precision = average_precision_score(y, y_hat)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.figure(figsize=(5, 4))
    plt.step(recall, precision, color='black', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='black', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title(data + ' Precision-Recall curve:\nAP={0:0.2f}'.format(average_precision))
    plt.savefig(dir_hps.save_dir + dir_hps.store_dir + '/prCurve_' + data + '.pdf', bbox_inches="tight")
