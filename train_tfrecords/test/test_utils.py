
import train_tfrecords.utils as utils
import os

def test_find_training_data():

    start_date = '2019.10.18.00.00'
    train_period = -1 
    train_data_path = os.getcwd() + '/train_tfrecords/test/data/'

    res = utils.find_training_data(start_date,train_period,train_data_path)
    assert res == [os.getcwd() + '/train_tfrecords/test/data/train_data_2019-10-17T17:09:12.637045.tf']

