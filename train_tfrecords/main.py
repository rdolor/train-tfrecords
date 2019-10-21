"""
This is the main file that calls all the other required modules for training and testing.
 Check the hyperparameters (the flags) from config.py to control the training process.
"""

import logging
import os
import nni

import train_tfrecords.config as config
import train_tfrecords.utils as utils
import train_tfrecords.train as train

from train_tfrecords.config import args, model_hps, train_hps, data_pipeline_hps, dir_hps, Colnames
from train_tfrecords.utils import output2csv

import logging
#log = utils.create_logger()


def main():
    logging.info('================= Model params =================')
    logging.info('model name:            %s', args['model_name'])
    logging.info('model:                 %s', train_hps.model)
    logging.info('loss:                  %d', model_hps.loss)
    logging.info('learning_rate:         %f', model_hps.learning_rate)
    logging.info('lambda:                %f', model_hps.Lambda)
    logging.info('gamma:                 %f', model_hps.gamma)
    logging.info('beta:                  %f', model_hps.beta)
    logging.info('drop_rate:             %f', model_hps.drop_rate)
    logging.info('================= Training params ==============')
    logging.info('is_test:               %d', train_hps.is_test)
    logging.info('train_batch_size:      %d', data_pipeline_hps.batch_size)
    logging.info('number iterations:     %d', train_hps.num_training)
    logging.info('number iterations min: %d', train_hps.num_training_min)
    logging.info('saving directory:      %s', dir_hps.save_dir)
    logging.info('loading directory:     %s', dir_hps.load_dir)
    logging.info('storing directory:     %s', dir_hps.store_dir)
    logging.info('================================================')

    data_result = []
    if not os.path.exists(dir_hps.save_dir + dir_hps.store_dir):
        os.makedirs(dir_hps.save_dir + dir_hps.store_dir)

    # Start training
    tr = train.Training(data_pipeline_hps, model_hps, train_hps, dir_hps)
    tr_threshold, tr_cutpoint, training_time, train_counter = tr.train()

    if train_hps.is_test == 1:
        if train_hps.is_percentile_threshold:
            train_hps.add_hparam('ctr_percentile_threshold', tr_threshold)
        else:
            train_hps.add_hparam('ctr_percentile_threshold', tr_cutpoint)
        data_result = tr.test()

    # Store results of training into a csv file
    data_result = [args['model_name'], training_time] + data_result + [train_counter,args['batch_size'],args['loss'],args['learning_rate'],args['decay_rate'],args['Lambda'],args['embedding_units'],args['hidden_units']]

    output2csv(data_result, Colnames, dir_hps)
    if args['tuning'] == 1:
        nni.report_final_result(data_result[20]) # Main evaluation metric: average precision score

    logging.info('Done all...')
    os._exit(0)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()


