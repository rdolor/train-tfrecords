import logging
import os

from datetime import datetime, timedelta
from glob import glob


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