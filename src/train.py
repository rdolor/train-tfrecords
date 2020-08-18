
""" This module handles the training and evaluation of a model.

The training process includes:
    * building the model (selects which model to use)
    * creating the input pipeline (uses data_generator.py)
    * training by batches until early stopping criterion is met
    Options to control the hyperparameters can be found in rblogis_usr_train.py.
"""

import logging
import os
import timeit
import numpy as np
import tensorflow as tf
import nni
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils import table_metrics, calculate_confusion_cutpoint
from src.data_generator import DataPipeline
from src.config import args
from src.model.dnn import DNN
from src.model.lr import LR

from shutil import rmtree


class Training:
    def __init__(self, data_pipeline_hps, model_hps, train_hps, dir_hps):
        self.data_pipeline_hps = data_pipeline_hps
        self.model_hps = model_hps
        self.train_hps = train_hps
        self.dir_hps = dir_hps

    def train(self):
        """Performs training of the model.
        :param data_pipeline_hps: input pipeline hyperparameter class
        :param model_hps: model hyperparameter class
        :param train_hps: training hyperparameter class
        :param dir_hps: model saving path class
        :return: * threshold_avg: float; This is the threshold that will be used to divide the probabilities into positive or negative class.
            Needed for the computation of the evaluation metrics.
        """
        tf.reset_default_graph()
        start_path = self.dir_hps.save_dir
        final_path = os.path.join(start_path, self.dir_hps.load_dir)
        builder_save_path = os.path.join(start_path, self.dir_hps.builder_save_dir)
        if not os.path.isdir(final_path):
            os.makedirs(final_path)
        if not os.path.isdir(builder_save_path):
            os.makedirs(builder_save_path)
        # Build graph model
        model = self.build_model()
        model.core_builder()

        # Start session
        sess_config = tf.ConfigProto(device_count={"CPU": self.data_pipeline_hps.num_cores},
                                     inter_op_parallelism_threads=24,
                                     intra_op_parallelism_threads=24)
        if self.train_hps.has_gpu == 1:
            logging.info('Controlling the use of GPU')
            sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)

        model_file = None
        if self.dir_hps.save_dir:
            model_file = tf.train.latest_checkpoint(self.dir_hps.save_dir + self.dir_hps.load_dir)
        if model_file:
            logging.info('Restoring from: %s', model_file)
            saver.restore(sess, model_file)

        # Construct data loader
        start_time_data_prep = timeit.default_timer()
        batch_data = DataPipeline(self.data_pipeline_hps)
        batch_data.build_train_data_tensor()
        batch_data.build_valid_data_tensor()
        stop_time_data_prep = timeit.default_timer()
        logging.info('Data prep time: %f secs', stop_time_data_prep - start_time_data_prep)

        # Prepare early stopping 
        self.earlystop_bool = False
        self.earlystop_not_improved = 0
        self.earlystop_frequency = 100000

        train_loss = []
        valid_loss = [0, 0]

        self.threshold_list = []
        self.threshold_buffer = [0] * self.train_hps.num_threshold_buffer

        self.cutpoint_list = []
        self.cutpoint_buffer = [0] * self.train_hps.num_threshold_buffer

        threshold_avg, cutpoint_avg = 0,0

        # Start of training
        start_time_training = timeit.default_timer()

        train_counter = 0
        while train_counter < self.train_hps.num_training:
            tr_xs, tr_ys = batch_data.get_train_next()
            if len(np.where(tr_ys == 1)[0]) > 0:
                train_counter += 1
                tr_loss, tr_y_hat = self.one_step(sess, model, tr_xs, tr_ys)
                
                if (train_counter <= 5)|(train_counter % self.train_hps.print_train_iter == 0):
                    self.print_result(train_counter, tr_loss, valid_loss)

                    if args['tuning'] == 1:
                        nni.report_intermediate_result(average_precision_score(tr_ys,tr_y_hat))

                if (train_counter % self.train_hps.save_model) == 0:
                    self.save_model(sess, saver)

                if (train_counter % min(self.earlystop_frequency, self.train_hps.validation_frequency)) == 0:
                    threshold_avg, cutpoint_avg = self.compute_cutpoint(tr_ys, tr_y_hat)
                    train_loss.append(tr_loss)
                    vloss = []
                    validation_counter = 0
                    while validation_counter < self.train_hps.validation_length:
                        v_xs, v_ys = batch_data.get_valid_next()
                        if len(np.where(v_ys == 1)[0]) > 0:
                            v_loss = self.one_validation(sess, model, v_xs, v_ys)
                            vloss.append(v_loss)
                            validation_counter += 1
                    valid_loss.append(np.mean(vloss))
                    if train_counter > self.train_hps.num_training_min:
                        self.check_earlystop(valid_loss)

            if self.earlystop_bool:
                logging.info("Early Stopping...")
                break

        self.save_model(sess, saver)
        np.save(self.dir_hps.save_dir + self.dir_hps.store_dir + '/threshold_avg.npy', (threshold_avg))
        np.save(self.dir_hps.save_dir + self.dir_hps.store_dir + '/cutpoint_avg.npy', (cutpoint_avg))

        self.builder_savedmodel(sess, graph=tf.get_default_graph())
        # self.freeze_savedmodel(sess)

        sess.close()
        batch_data.close()
        del batch_data

        stop_time_training = timeit.default_timer()
        training_time = stop_time_training - start_time_training
        logging.info('Training time: %f secs', training_time)
        logging.info('threshold_avg = %f, cutpoint_avg = %f', threshold_avg, cutpoint_avg)
        return threshold_avg, cutpoint_avg, training_time, train_counter

    def test(self):
        """Performs the evaluation of the trained model.
        """
        tf.reset_default_graph()
        start_path = self.dir_hps.save_dir
        final_path = os.path.join(start_path, self.dir_hps.load_dir)

        if not os.path.isdir(final_path):
            os.makedirs(final_path)

        # Build graph model
        model = self.build_model()
        model.core_builder()

        # Start a session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)

        model_file = None
        if self.dir_hps.save_dir:
            model_file = tf.train.latest_checkpoint(self.dir_hps.save_dir + self.dir_hps.store_dir)
        if model_file:
            logging.info('Restoring from: %s', model_file)
            saver.restore(sess, model_file)

        # Construct data loader 
        batch_data = DataPipeline(self.data_pipeline_hps)
        batch_data.build_train_data_tensor()
        batch_data.build_valid_data_tensor()
        batch_data.build_test_data_tensor()

        self.metrics = {
                'train': {'all_auc':[],'all_ys':[],'all_y_hat':[],'batch_auc':[],'ctr':[],'loss':[],'all_apr':[],'batch_apr':[],'prec':[],'rec':[]},
                'valid': {'all_auc':[],'all_ys':[],'all_y_hat':[],'batch_auc':[],'ctr':[],'loss':[],'all_apr':[],'batch_apr':[],'prec':[],'rec':[]},
                'test' : {'all_auc':[],'all_ys':[],'all_y_hat':[],'batch_auc':[],'ctr':[],'loss':[],'all_apr':[],'batch_apr':[],'prec':[],'rec':[]}}

        metrics_all = []

        for dataset in ['train', 'valid', 'test']:
            logging.info('====================== ' + dataset + ' ======================')
            test_counter = 0
            while test_counter < self.train_hps.test_length:
                if dataset == 'train':
                    xs, ys = batch_data.get_train_next()
                elif dataset == 'valid':
                    xs, ys = batch_data.get_valid_next()
                else:
                    xs, ys = batch_data.get_test_next()

                if len(np.where(ys == 1)[0]) > 0:
                    loss, y_hat = self.one_validation(sess, model, xs, ys, include_prediction=True)
                    self.append_metrics(dataset, ys, y_hat, loss)
                    test_counter += 1

            ctr, loss, prec, rec, batch_auc, all_auc, batch_apr, all_apr = self.compute_mean_metrics(dataset)
            metrics_all.extend([all_auc, batch_auc,all_apr, batch_apr, ctr, loss, prec, rec])
            self.save_evaluation_metrics(dataset)

        sess.close()
        batch_data.close()
        del batch_data

        return metrics_all

    def build_model(self):
        """ Constructs the static TF model."""
        if self.train_hps.model == 'DNN':
            assert self.data_pipeline_hps.to_ohe==0,"Error:to_ohe = 1. If model='DNN', then data_pipeline_hps.to_ohe should be 0."
            model = DNN(self.model_hps)
        elif self.train_hps.model == 'LR':
            assert self.data_pipeline_hps.to_ohe==1,"Error:to_ohe = 0. If model='LR', then data_pipeline_hps.to_ohe should be 1."
            model = LR(self.model_hps)
        else:
            logging.info('Add other models: %s', self.train_hps.model)
            model = None
        return model

    def print_result(self, iteration, tr_loss, valid_loss):
        """ Prints the training and validation loss. """
        if len(valid_loss) <= 1:
            logging.info('Iteration: %d Training Loss: %f', iteration, np.round(tr_loss, 4))
        else:
            logging.info('Iteration: %d Training Loss: %f Validation Loss: %f vloss diff: %f',
                         iteration, np.round(tr_loss, 4), np.round(valid_loss[-1], 4),
                         np.round(valid_loss[-2] - valid_loss[-1], 4))


    def save_model(self, sess, saver):
        """ Saves the current model."""
        model_file = self.dir_hps.save_dir + self.dir_hps.store_dir + '/model'
        saver.save(sess, model_file)

    def builder_savedmodel(self, sess, graph):
        """ Saves a builder savedmodel. This is the format needed to use TFJS."""

        export_dir = self.dir_hps.save_dir + self.dir_hps.builder_save_dir + '/saved_model'
        if os.path.isdir(export_dir):
            rmtree(export_dir)

        if self.train_hps.model == 'DNN': #weekday', 'region','city', 'adexchange', 'slotformat
            phase_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("phase:0"))
            weekday_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("weekday:0"))
            region_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("region:0"))
            city_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("city:0"))
            adexchange_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("adexchange:0"))
            slotformat_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("slotformat:0"))

            hour_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("hour:0"))
            slotwidth_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("slotwidth:0"))
            slotheight_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("slotheight:0"))
            slotvisibility_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("slotvisibility:0"))
            slotprice_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("slotprice:0"))

            usertag_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("usertag:0"))

            outputs_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("prediction_node:0"))
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={
                                'phase': phase_classes,
                                'weekday': weekday_classes,
                                'region': region_classes,
                                'city': city_classes,
                                'adexchange_code': adexchange_classes,
                                'slotformat_code': slotformat_classes,
                                
                                'hour': hour_classes,
                                'slotwidth': slotwidth_classes,
                                'slotheight': slotheight_classes,
                                'slotvisibility': slotvisibility_classes,
                                'slotprice': slotprice_classes,
                                
                                'usertag': usertag_classes
                            },
                            outputs={'outputs': outputs_classes},
                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
            
            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
            builder.add_meta_graph_and_variables(sess,
                                                [tf.saved_model.tag_constants.SERVING],
                                                signature_def_map={'serving_default': prediction_signature, }
                                                )
            builder.save()
            logging.info("SavedModel graph built")

        elif self.train_hps.model == 'LR':
            inputs_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("input_node:0"))
            outputs_classes = tf.saved_model.utils.build_tensor_info(graph.get_tensor_by_name("prediction_node:0"))
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={'inputs': inputs_classes},
                            outputs={'outputs': outputs_classes},
                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        else:
            logging.info('Add other models: %s', self.train_hps.model)


    def freeze_savedmodel(self, sess):
        from tensorflow.python.tools import freeze_graph
        from tensorflow.python.saved_model import tag_constants

        export_dir = self.dir_hps.save_dir + self.dir_hps.builder_save_dir + '/saved_model'
        if os.path.isdir(export_dir):
            os.removedirs(export_dir)

        output_graph_filename = os.path.join(export_dir, "freezed_model.pb")
        output_node_names = "prediction_node"
        initializer_nodes = ""

        freeze_graph.freeze_graph(
            input_saved_model_dir=export_dir,
            output_graph=output_graph_filename,
            saved_model_tags=tag_constants.SERVING,
            output_node_names=output_node_names,
            initializer_nodes=initializer_nodes,

            input_graph=None,
            input_saver=False,
            input_binary=False,
            input_checkpoint=None,
            restore_op_name=None,
            filename_tensor_name=None,
            clear_devices=False,
            input_meta_graph=False,
        )
        logging.info("SavedModel graph freezed")

    def compute_cutpoint(self, tr_ys, tr_y_hat):
        """ Computes the threshold where to cut the probabilites. """
        cutpoint_avg = 0
        y_hat_percentile = np.percentile(tr_y_hat, q=np.arange(0, 100, 10))
        threshold = y_hat_percentile[self.train_hps.percentile_threshold]
        self.threshold_list.append(threshold)
        self.threshold_buffer = self.threshold_buffer[1:] + [self.threshold_list[-1]]
        threshold_avg = np.mean(self.threshold_buffer)
        try:
            _, _, cutpoint = calculate_confusion_cutpoint(tr_ys, tr_y_hat, "IU")
            self.cutpoint_list.append(cutpoint)
            self.cutpoint_buffer = self.cutpoint_buffer[1:] + [self.cutpoint_list[-1]]
            self.cutpoint_buffer = [i for i in self.cutpoint_buffer if i is not None]
            if len(self.cutpoint_buffer) > 0:
                cutpoint_avg = np.mean(self.cutpoint_buffer)
        except ValueError:
            cutpoint_avg = -1
        return threshold_avg, cutpoint_avg

    def one_step(self, sess, model, tr_xs, tr_ys):
        """ Train the model for one step.
        """

        epsilon = 1e-7
        num_rows = np.shape(tr_ys)[0]
        pos = sum(tr_ys)
        neg = num_rows - pos
        pos_weight = neg / (pos + epsilon)
        neg_weight = 1.0
        batch_weights = [pos_weight, neg_weight]
        
        if self.train_hps.model == 'DNN':
            _, tr_loss, tr_y_hat = sess.run([model.train_op, model.loss_op, model.y_prob],
                                            feed_dict={
                                                model.phase: 1,
                                                model.y:tr_ys.reshape(num_rows, 1),
                                                model.drop_rate: self.model_hps.drop_rate,

                                                model.features_dict['weekday'][2]: tr_xs['weekday'].reshape(num_rows, 1),
                                                model.features_dict['region'][2]: tr_xs['region'].reshape(num_rows, 1),
                                                model.features_dict['city'][2]:tr_xs['city'].reshape(num_rows, 1),
                                                model.features_dict['adexchange'][2]: tr_xs['adexchange'].reshape(num_rows, 1),
                                                model.features_dict['slotformat'][2]: tr_xs['slotformat'].reshape(num_rows, 1),
    
                                                model.hour: tr_xs['hour'].reshape(num_rows, 1),
                                                model.slotwidth: tr_xs['slotwidth'].reshape(num_rows, 1),
                                                model.slotheight: tr_xs['slotheight'].reshape(num_rows, 1),
                                                model.slotvisibility: tr_xs['slotvisibility'].reshape(num_rows, 1),
                                                model.slotprice:tr_xs['slotprice'].reshape(num_rows, 1),
                
                                                model.usertag:tr_xs['usertag'].reshape(num_rows, -1),

                                                # others
                                                model.class_weights: batch_weights,
                                                model.loss_cond: self.model_hps.loss
                                            })

        else:
            tr_xs = tr_xs.reshape(num_rows, self.model_hps.num_features)
            tr_ys = tr_ys.reshape(num_rows, 1)
            _, tr_loss, tr_y_hat = sess.run([model.train_op, model.loss_op, model.y_prob],
                                            feed_dict={ model.phase: 1,
                                                        model.x: tr_xs,
                                                        model.y: tr_ys,
                                                        model.class_weights: batch_weights,
                                                        model.loss_cond: self.model_hps.loss})


        return tr_loss, tr_y_hat

    def one_validation(self, sess, model, xs, ys, include_prediction=False):
        """ Computes loss and prediction for one batch of data."""
        epsilon = 1e-7
        num_rows = np.shape(ys)[0]
        pos = sum(ys)
        neg = num_rows - pos
        pos_weight = neg / (pos + epsilon)
        neg_weight = 1.0
        batch_weights = [pos_weight, neg_weight]

        if self.train_hps.model == 'DNN':
            model_feed_dict={

                model.y:ys.reshape(num_rows, 1),
                model.phase: 0,

                model.features_dict['weekday'][2]: xs['weekday'].reshape(num_rows, 1),
                model.features_dict['region'][2]: xs['region'].reshape(num_rows, 1),
                model.features_dict['city'][2]:xs['city'].reshape(num_rows, 1),
                model.features_dict['adexchange'][2]: xs['adexchange'].reshape(num_rows, 1),
                model.features_dict['slotformat'][2]: xs['slotformat'].reshape(num_rows, 1),

                model.hour: xs['hour'].reshape(num_rows, 1),
                model.slotwidth: xs['slotwidth'].reshape(num_rows, 1),
                model.slotheight: xs['slotheight'].reshape(num_rows, 1),
                model.slotvisibility: xs['slotvisibility'].reshape(num_rows, 1),
                model.slotprice:xs['slotprice'].reshape(num_rows, 1),

                model.usertag:xs['usertag'].reshape(num_rows, -1),
                                            
                model.class_weights: batch_weights,
                model.loss_cond: self.model_hps.loss

            }
        
        else:
            xs = xs.reshape(num_rows, self.model_hps.num_features)
            ys = ys.reshape(num_rows, 1)
            model_feed_dict = { model.phase: 0,
                                model.x: xs,
                                model.y: ys,
                                model.class_weights: batch_weights,
                                model.loss_cond: self.model_hps.loss}

        if include_prediction:
            loss, y_hat = sess.run([model.loss_op, model.y_prob],feed_dict=model_feed_dict)
            return loss, y_hat

        else:
            loss = sess.run(model.loss_op,feed_dict=model_feed_dict)
            return loss

    def check_earlystop(self, valid_loss):
        """ Performs checking of early stop.
        Turn ON early stop checking only when the minimum number of training is reached.
        After which, should make more frequent validation checking.
        """
        self.earlystop_frequency = self.train_hps.earlystop_check_frequency
        if abs(valid_loss[-1] - valid_loss[-2]) <= self.train_hps.valid_loss_delta:
            self.earlystop_not_improved += 1
        else:
            self.earlystop_not_improved = 0
        logging.info('earlystop_not_improved = %d', self.earlystop_not_improved)
        if self.earlystop_not_improved >= self.train_hps.earlystop_duration:
            self.earlystop_bool = True

    def append_metrics(self, dataset, ys, y_hat, loss):
        """ During testing, this appends the computed stats to the list
        of metrics of a particular dataset (train/valid/test).
        """
        predicted_values1 = y_hat
        predicted_class1 = np.zeros(predicted_values1.shape)
        predicted_class1[predicted_values1 > self.train_hps.ctr_percentile_threshold] = 1
        overall_ctr = sum(predicted_class1) / len(predicted_class1)

        try:
            self.metrics[dataset]['all_ys'] = np.concatenate((self.metrics[dataset]['all_ys'], ys.reshape(-1)), axis=0)
            self.metrics[dataset]['all_y_hat'] = np.concatenate((self.metrics[dataset]['all_y_hat'], y_hat.reshape(-1)),
                                                                axis=0)
        except:
            self.metrics[dataset]['all_ys'] = self.metrics[dataset]['all_ys'].reshape(-1)
            self.metrics[dataset]['all_y_hat'] = self.metrics[dataset]['all_y_hat'].reshape(-1)

        recall, precision, _ = table_metrics(ys, predicted_class1)
        self.metrics[dataset]['ctr'].append(overall_ctr)
        self.metrics[dataset]['loss'].append(loss)
        self.metrics[dataset]['prec'].append(precision)
        self.metrics[dataset]['rec'].append(recall)
        self.metrics[dataset]['batch_auc'].append(roc_auc_score(ys, y_hat))
        self.metrics[dataset]['batch_apr'].append(average_precision_score(ys, y_hat))


    def compute_mean_metrics(self, dataset):
        """Computes the average of the evaluation metrics so far."""
        ctr = np.nanmean(self.metrics[dataset]['ctr'])
        loss = np.nanmean(self.metrics[dataset]['loss'])
        prec = np.nanmean(self.metrics[dataset]['prec'])
        rec = np.nanmean(self.metrics[dataset]['rec'])
        batch_auc = np.nanmean(self.metrics[dataset]['batch_auc'])
        batch_apr = np.nanmean(self.metrics[dataset]['batch_apr'])

        try:
            all_auc = roc_auc_score(self.metrics[dataset]['all_ys'].T, self.metrics[dataset]['all_y_hat'].T)
            all_apr = average_precision_score(self.metrics[dataset]['all_ys'].T, self.metrics[dataset]['all_y_hat'].T)
        except Exception as err:
            logging.info('[Error] Concatenated data AUC: %s', str(err))
            all_auc = -1
            all_apr = -1

        logging.info('All data AUC   : %f', np.round(all_auc, 4))
        logging.info('Batch AUC      : %f', np.round(batch_auc, 4))
        logging.info('All data Avg PR: %f', np.round(all_apr, 4))
        logging.info('Batch Avg PR   : %f', np.round(batch_apr, 4))
        logging.info('Predicted CTR  : %f', np.round(ctr, 4))
        logging.info('Loss           : %f', np.round(loss, 4))
        logging.info('Precision      : %f', np.round(prec, 4))
        logging.info('Recall         : %f', np.round(rec, 4))

        return ctr, loss, prec, rec, batch_auc, all_auc, batch_apr, all_apr

    def save_evaluation_metrics(self, dataset):
        """Stores the evaluation metrics in numpy format."""
        np.save(self.dir_hps.save_dir + self.dir_hps.store_dir + '/' + dataset + '_evalmetrics.npy', \
                (self.metrics[dataset]['ctr'], self.metrics[dataset]['loss'],
                 self.metrics[dataset]['prec'], self.metrics[dataset]['rec'],
                 self.metrics[dataset]['batch_auc'], self.metrics[dataset]['all_auc'],
                 self.metrics[dataset]['batch_apr'], self.metrics[dataset]['all_apr']
                 ))

        np.save(self.dir_hps.save_dir + self.dir_hps.store_dir + '/all_' + dataset + 'ys.npy',
                (self.metrics[dataset]['all_ys']))
        np.save(self.dir_hps.save_dir + self.dir_hps.store_dir + '/all_' + dataset + 'y_hat.npy',
                (self.metrics[dataset]['all_y_hat']))
