
from train_tfrecords.data_generator import DataPipeline
from train_tfrecords.config import data_pipeline_hps, model_hps

import os
import numpy as np

#print(data_pipeline_hps.train_data_path)
#print(data_pipeline_hps.data_file)

def test_data_pipeline_ohe():
    #data_pipeline_hps['data_file'] = [os.getcwd() + '/train_ohe_tf/test/data/train_data_2019-10-17T17:09:12.637045.tf']
    data_pipeline_hps.data_file = [os.getcwd() + '/train_tfrecords/test/data/train_data_2019-10-17T17:09:12.637045.tf']
    data_pipeline_hps.batch_size = 10
    data_pipeline_hps.to_ohe = 1
    
    batch_data = DataPipeline(data_pipeline_hps)
    batch_data.build_train_data_tensor()
    batch_data.build_valid_data_tensor()
    batch_data.build_test_data_tensor()

    tr_xs, _ = batch_data.get_train_next()
    v_xs, _  = batch_data.get_valid_next()
    ts_xs, _ = batch_data.get_test_next()

    assert np.shape(tr_xs)[2] == model_hps.num_features
    assert np.shape(v_xs)[2] == model_hps.num_features
    assert np.shape(ts_xs)[2] == model_hps.num_features
