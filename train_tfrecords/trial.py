
import tensorflow as tf
import timeit
import numpy as np
import os

from train_tfrecords.data_generator import DataPipeline
from train_tfrecords.config import data_pipeline_hps, dir_hps, train_hps
import train_tfrecords.utils as utils


log = utils.create_logger()

tf.reset_default_graph()
start_path = dir_hps.save_dir
final_path = os.path.join(start_path, dir_hps.load_dir)
builder_save_path = os.path.join(start_path, dir_hps.builder_save_dir)
if not os.path.isdir(final_path):
    os.makedirs(final_path)
if not os.path.isdir(builder_save_path):
    os.makedirs(builder_save_path)


# Start session
sess_config = tf.ConfigProto(device_count={"CPU": data_pipeline_hps.num_cores},
                                inter_op_parallelism_threads=24,
                                intra_op_parallelism_threads=24)
if train_hps.has_gpu == 1:
    log.info('Controlling the use of GPU')
    sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())

# Construct data loader
start_time_data_prep = timeit.default_timer()
batch_data = DataPipeline(data_pipeline_hps)
batch_data.build_train_data_tensor()
batch_data.build_valid_data_tensor()

stop_time_data_prep = timeit.default_timer()
log.info('Data prep time: %f secs', stop_time_data_prep - start_time_data_prep)

tr_xs, tr_ys = batch_data.get_train_next()
print(np.shape(tr_xs["usertag"]), np.shape(tr_ys)) # ohe: xs=(10000, 1, 456) ys=(10000, 1)
print(np.shape(tr_xs["usertag"]), np.shape(tr_ys)) # NOT (10000, 45) (10000, 1)
