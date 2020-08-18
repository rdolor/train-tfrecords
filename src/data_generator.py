"""
This module handles the creation of input pipeline that will be used for training the model.

Input: TFrecords
Output: Batches of training data
"""

import tensorflow as tf
import random
import logging

from src.config import NUM_WEEKDAY, NUM_REGION, NUM_CITY, NUM_ADEXCHANGE, NUM_SLOTFORMAT, NUM_USERTAG
from src.config import features_dtype_int, numerical_features

#from config import NUM_WEEKDAY, NUM_REGION, NUM_CITY, NUM_adexchange, NUM_SLOTFORMAT, NUM_USERTAG
#from config import features_dtype_int, numerical_features


class DataPipeline():
    """ Given a list of location of TFrecord files, this creates the pipeline for training .
    The options to control the parameters can be found in 'input_pipeline_hps' in the main file (rblogis_usr_train.py).

    Example usage:
        batch_data = inputPipeline(input_pipeline_hps)
        batch_data.build_train_data_tensor()
        batch_data.build_valid_data_tensor()
    """
    
    def __init__(self, input_pipeline_hps):
        self.input_pipeline_hps = input_pipeline_hps
        self.graph = tf.Graph()

        sess_config = tf.ConfigProto(device_count={"CPU":self.input_pipeline_hps.num_cores},
                                 inter_op_parallelism_threads=24,
                                 intra_op_parallelism_threads=24)
        if self.input_pipeline_hps.has_gpu == 1:
            logging.info('Controlling the use of GPU')
            sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        
        # Build the full_data tensor
        with self.graph.as_default():
            self.__build_data_tensor()
        

    def build_train_data_tensor(self):
        with self.graph.as_default():
            train_dataset = self.train_dataset.repeat()
            train_dataset = train_dataset.batch(self.input_pipeline_hps.batch_size)
            train_dataset = train_dataset.prefetch(self.input_pipeline_hps.prefetch_size)
            train_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            train_init_op = train_iterator.make_initializer(train_dataset)
            self.train_next_batch = train_iterator.get_next()

            self.sess.run(train_init_op)
            
    def build_valid_data_tensor(self):
        with self.graph.as_default():
            valid_dataset = self.valid_dataset.repeat()
            valid_dataset = valid_dataset.batch(self.input_pipeline_hps.batch_size) 
            valid_dataset = valid_dataset.prefetch(self.input_pipeline_hps.prefetch_size)
            valid_iterator = tf.data.Iterator.from_structure(valid_dataset.output_types, valid_dataset.output_shapes)
            valid_init_op = valid_iterator.make_initializer(valid_dataset)
            self.valid_next_batch = valid_iterator.get_next()

            self.sess.run(valid_init_op)
            
    def build_test_data_tensor(self):
        with self.graph.as_default():
            test_dataset = self.test_dataset.repeat()
            test_dataset = test_dataset.batch(self.input_pipeline_hps.batch_size) 
            test_dataset = test_dataset.prefetch(self.input_pipeline_hps.prefetch_size)
            test_iterator = tf.data.Iterator.from_structure(test_dataset.output_types, test_dataset.output_shapes)
            test_init_op = test_iterator.make_initializer(test_dataset)
            self.test_next_batch = test_iterator.get_next()

            self.sess.run(test_init_op)           


    def __build_data_tensor(self):
        
        DATASET_SIZE = 0
        for fn in self.input_pipeline_hps.data_file:
            for _ in tf.python_io.tf_record_iterator(fn):
                 DATASET_SIZE += 1
        logging.info('TOTAL DATASET_SIZE = %d', DATASET_SIZE)
        
        random.seed(self.input_pipeline_hps.random_seed)
        random.shuffle(self.input_pipeline_hps.data_file)

        def extract_tfrecords(data_record):
            
            features =  {}  
            for int_feature in features_dtype_int:
                features[int_feature] = tf.FixedLenFeature([1], tf.int64)
                
            features["usertag"] = tf.VarLenFeature(tf.int64)
            
            sample = tf.parse_single_example(data_record, features)
            
            sample["usertag"] = tf.sparse_tensor_to_dense(sample["usertag"], default_value=0)
            sample["usertag"] = tf.reduce_sum(tf.one_hot(sample["usertag"], depth=NUM_USERTAG),axis=0)

            y = sample["click"]

            if self.input_pipeline_hps.to_ohe:
                
                sample["usertag"] = tf.cast(sample["usertag"], tf.int64)
                
                ohe_weekday    = tf.cast(tf.one_hot(sample["weekday"] , depth=NUM_WEEKDAY,   on_value=1, axis=1), dtype=tf.int64)
                ohe_region     = tf.cast(tf.one_hot(sample["region"] , depth=NUM_REGION,          on_value=1, axis=1), dtype=tf.int64)
                ohe_city       = tf.cast(tf.one_hot(sample["city"]   , depth=NUM_CITY,          on_value=1, axis=1), dtype=tf.int64)
                ohe_adexchange   = tf.cast(tf.one_hot(sample["adexchange"] , depth=NUM_ADEXCHANGE,         on_value=1, axis=1), dtype=tf.int64)
                ohe_slotformat = tf.cast(tf.one_hot(sample["slotformat"] , depth=NUM_SLOTFORMAT,         on_value=1, axis=1), dtype=tf.int64)

                sample["usertag"] = tf.reshape(sample["usertag"],[1,-1])
                
                ohe_weekday    = tf.reshape(ohe_weekday,[tf.shape(ohe_weekday)[0],-1])
                ohe_region     = tf.reshape(ohe_region,[tf.shape(ohe_region)[0],-1])
                ohe_city       = tf.reshape(ohe_city,[tf.shape(ohe_city)[0],-1])
                ohe_adexchange   = tf.reshape(ohe_adexchange,[tf.shape(ohe_adexchange)[0],-1])
                ohe_slotformat = tf.reshape(ohe_slotformat,[tf.shape(ohe_slotformat)[0],-1])
                
                x_ohe = tf.concat([sample["usertag"],
                        ohe_weekday, ohe_region, ohe_city, ohe_adexchange, ohe_slotformat],axis=1)

                x_ohe = tf.cast(x_ohe, tf.float32)

                for numerical_feature in numerical_features:
                    sample[numerical_feature] = tf.cast(sample[numerical_feature], tf.float32)
                    sample[numerical_feature] = tf.reshape(sample[numerical_feature], [tf.shape(sample[numerical_feature])[0],-1])
                ['hour', 'slotwidth','slotheight', 'slotvisibility', 'slotprice']

                x_ohe = tf.concat([x_ohe,
                        sample["hour"],sample["slotwidth"],sample["slotheight"],
                        sample["slotvisibility"],sample["slotprice"]],
                        axis=1)
                
                return (x_ohe, y)  # x_ohe: array, y:int

            else:
                del sample["click"]

                return (sample, y) # sample: dict, y: int 
        
        full_dataset = tf.data.TFRecordDataset(self.input_pipeline_hps.data_file)
        
        if self.input_pipeline_hps.is_test:
            test_ratio = 1.0 - self.input_pipeline_hps.train_ratio - self.input_pipeline_hps.valid_ratio 
            train_size = int(self.input_pipeline_hps.train_ratio * DATASET_SIZE)
            valid_size = int(self.input_pipeline_hps.valid_ratio * DATASET_SIZE)
            test_size  = int(test_ratio * DATASET_SIZE)
            logging.info('train_size: %d valid_size: %d test_size: %d', train_size, valid_size, test_size)
            
            full_dataset = full_dataset.shuffle(buffer_size=DATASET_SIZE)
            full_dataset = full_dataset.map(extract_tfrecords,
                                num_parallel_calls=self.input_pipeline_hps.num_cores)

            self.train_dataset = full_dataset.take(train_size)
            self.test_dataset = full_dataset.skip(train_size)
            self.valid_dataset = self.test_dataset.skip(valid_size)
            self.test_dataset = self.test_dataset.take(test_size)
            
        else:
            train_size = int(self.input_pipeline_hps.train_ratio * DATASET_SIZE)
            valid_size = DATASET_SIZE - train_size
            logging.info('train_size: %d valid_size: %d', train_size, valid_size)
            
            full_dataset = full_dataset.shuffle(buffer_size=DATASET_SIZE)
            full_dataset = full_dataset.map(extract_tfrecords, num_parallel_calls=self.input_pipeline_hps.num_cores)
            self.train_dataset = full_dataset.take(train_size)
            self.valid_dataset = full_dataset.skip(train_size)

            
    def get_train_next(self):
        with self.graph.as_default():
            return self.sess.run(self.train_next_batch)
    
    def get_valid_next(self):
        with self.graph.as_default():
            return self.sess.run(self.valid_next_batch)
        
    def get_test_next(self):
        with self.graph.as_default():
            return self.sess.run(self.test_next_batch)
        
    def __del__(self):
        self.close()
        
    def close(self):
        self.sess.close()