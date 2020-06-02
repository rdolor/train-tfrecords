
""" This file contains the (hyper)parameters used to make the whole PyTrain run.

Data information:
feature = click,          shape = (10000, 1),  Unique count = 2,   min = 0,   max = 1
feature = weekday,        shape = (10000, 1),  Unique count = 1,   min = 4,   max = 4
feature = region,         shape = (10000, 1),  Unique count = 35,  min = 0,   max = 395
feature = city,           shape = (10000, 1),  Unique count = 359, min = 0,   max = 399
feature = adexchange,     shape = (10000, 1),  Unique count = 3,   min = 1,   max = 3
feature = slotformat,     shape = (10000, 1),  Unique count = 2,   min = 0,   max = 1
feature = hour,           shape = (10000, 1),  Unique count = 1,   min = 0,   max = 0
feature = slotwidth,      shape = (10000, 1),  Unique count = 6,   min = 160, max = 1000
feature = slotheight,     shape = (10000, 1),  Unique count = 4,   min = 90,  max = 600
feature = slotvisibility, shape = (10000, 1),  Unique count = 4,   min = 0,   max = 255
feature = slotprice,      shape = (10000, 1),  Unique count = 46,  min = 0,   max = 280
feature = usertag,        shape = (10000, 39), Unique count = 45,  min = -1,  max = 16706

python -m src.main
pytest train_tfrecords/test/test_utils.py
pytest train_tfrecords/test/test_data_generator.py

tuning: 
    cd train_tfrecords/tune
    nnictl create --config tune_config.yml --port <PORT_NUMBER>

build docker:
bash build.sh
docker run -it <IMAGE ID> bash
either:
    bash iter_train.sh; OR
    python -m src.main

changed:
-fixed DNN training
-added nni tuning
-added iter_train.sh
"""

import os

import logging
import tensorflow as tf
import nni
from configargparse import ArgParser
from src.utils import parse_date, find_training_data, find_latest_model_dir


# Features
features_dtype_int = ['click', 'weekday', 'region', 'city', 'adexchange', 'slotformat', 'hour', 'slotwidth',
                      'slotheight', 'slotvisibility', 'slotprice']
features_dtype_list = ['usertag']
all_features = features_dtype_int + features_dtype_list


categorical_features = ['weekday', 'region',
                        'city', 'adexchange', 'slotformat']
numerical_features = ['hour', 'slotwidth',
                      'slotheight', 'slotvisibility', 'slotprice']


# Categorical features = number of elements
NUM_WEEKDAY              = 8
NUM_REGION               = 396
NUM_CITY                 = 400
NUM_ADEXCHANGE           = 4
NUM_SLOTFORMAT           = 3
NUM_USERTAG              = 16707
NUM_CATEGORICAL_FEATURES = NUM_WEEKDAY+NUM_REGION + \
    NUM_CITY+NUM_ADEXCHANGE+NUM_SLOTFORMAT+NUM_USERTAG

DATA_PATH    = os.getcwd() + '/tests/data/'
START_DATE   = '2019.10.18.00.00'
TRAIN_PERIOD = -1


def ParseArgs():
    parser = ArgParser(default_config_files=[
                       os.getcwd() + '/src/initial_configurations/default'])

    # Core settings
    core_parse = parser.add_argument_group('Core setting')
    core_parse.add_argument('-s', '--start_date',         dest='start_date',
                            default=START_DATE,   type=str, help='Training start date')
    core_parse.add_argument('-p','--train_period',       dest='train_period',
                            default=TRAIN_PERIOD, type=int, help='Time period of training file is used')
    core_parse.add_argument('-n','--new_run',              dest='new_run',         default=1,
                             type=int, help='If the model checkpoint is erased to run new model')
    core_parse.add_argument('-l','--local_run',            dest='local_run',       default=1,
                             type=int, help='If the parameter JSON file is kept locally insteat to  redis')
    core_parse.add_argument('-nni','--tuning',               dest='tuning',          default=0,
                             type=int, help='Whether or not to peform hyper parameter tuning')

    # Data
    data_parse = parser.add_argument_group('Data setting')
    data_parse.add_argument('--train_data_path',    dest='train_data_path',
                            default=DATA_PATH,  type=str, help='Directory where the training files are located')
    
    data_parse.add_argument('--random_seed',        dest='random_seed',        default=8888,
                            type=int, help='Random seed used for shuffling the list of training files')
    data_parse.add_argument('--num_cores',          dest='num_cores',
                            default=24,         type=int, help='Number of CPU cores')
    data_parse.add_argument('--train_ratio',        dest='train_ratio',        default=0.7,
                            type=float, help='Fraction of data to be used for training')
    data_parse.add_argument('--valid_ratio',        dest='valid_ratio',        default=0.15,       type=float,
                            help='Fraction of data to be used for validation (only matters when there is a third dataset to be created for testing)')
    data_parse.add_argument('--batch_size',         dest='batch_size',
                            default=32,         type=int, help='Number of examples per batch')
    data_parse.add_argument('--prefetch_size',      dest='prefetch_size',      default=1,
                            type=int, help='Number of batches to be prepared in queue')

    # Model
    model_parse = parser.add_argument_group('Model setting')
    model_parse.add_argument('--model',                dest='model',               default='DNN',
                             type=str,   help='Select the model to train e.g. DNN; note that this version only has DNN')
    model_parse.add_argument('--loss',            dest='loss',              default=40,
                             type=int,   help="Setting of loss function '10','11','12','20','21','22','30','31','32','40'")
    model_parse.add_argument('--hidden_units',    dest='hidden_units',      default=[
                             128, 64],      type=int,   nargs='+', help='List containing the number of hidden units to use for each hidden layer')
    model_parse.add_argument('--learning_rate',   dest='learning_rate',
                             default=0.001,         type=float, help='Learning rate of updating gradient')
    model_parse.add_argument('--decay_step',      dest='decay_step',
                             default=100,           type=int,   help='Decay step')
    model_parse.add_argument('--decay_rate',      dest='decay_rate',        default=0.98,
                             type=float, help='Decay rate for exponential decay of learning rate')
    model_parse.add_argument('--Lambda',          dest='Lambda',            default=0.25,
                             type=float, help='Lambda for L2,L1 regularization; alpha for focal loss')
    model_parse.add_argument('--gamma',           dest='gamma',
                             default=2.,            type=float, help='parameter for focal loss')
    model_parse.add_argument('--beta',            dest='beta',
                             default=1.,            type=float, help='Regularization parameter')
    model_parse.add_argument('--drop_rate',       dest='drop_rate',
                             default=0.5,            type=float, help='dropout rate')
    model_parse.add_argument('--embedding_units', dest='embedding_units',   default=[1, 35, 359, 3, 2], type=int, nargs='+',
                             help='List containing the number of embedding units to use for features (in order): [weekday, region, city, adexchange, slotformat]; this replaces the one hot encoding')
    model_parse.add_argument('--embedding_units_ohe', dest='embedding_units_ohe',   default=[
                             45], type=int, nargs='+', help='List containing the number of embedding units to use for OHE features (in order): usertag')

    # Training
    train_parse = parser.add_argument_group('Training hyperparameters')
    train_parse.add_argument('--has_gpu',              dest='has_gpu',
                             default=0,     type=int,   help='1 if GPU is present, else 0')
    train_parse.add_argument('--is_test',              dest='is_test',             default=0,
                             type=int,   help='1 if the trained model will be evaluated, else 0')
    train_parse.add_argument('--num_epochs_min',       dest='num_epochs_min',
                             default=100,   type=float,   help='Minimum number of training epochs')
    train_parse.add_argument('--num_epochs',           dest='num_epochs',
                             default=101,   type=float,   help='Number of total training epochs')
    train_parse.add_argument('--validation_length',    dest='validation_length',   default=100,
                             type=int,   help='In one validation, how many number of batches to use')
    train_parse.add_argument('--test_length',          dest='test_length',
                             default=100,   type=int,   help='In one test, how many number of batches to use')
    train_parse.add_argument('--earlystop_check_frequency', dest='earlystop_check_frequency',
                             default=10,     type=int,   help='earlystop_check_frequency')
    train_parse.add_argument('--earlystop_duration',       dest='earlystop_duration',
                             default=10,     type=int,   help='earlystop_duration')
    train_parse.add_argument('--valid_loss_delta',         dest='valid_loss_delta',
                             default=0.0001, type=float, help='valid_loss_delta')
    train_parse.add_argument('--num_threshold_buffer',     dest='num_threshold_buffer',
                             default=3,      type=int,   help='num_threshold_buffer')
    train_parse.add_argument('--percentile_threshold',     dest='percentile_threshold',
                             default=8,      type=int,   help='percentile_threshold')

    # Directory paths
    dir_parse = parser.add_argument_group('Directory paths')
    dir_parse.add_argument('--save_dir',         dest='save_dir',
                           default='./Outputs/',        type=str, help='Directory to save model directories')
    dir_parse.add_argument('--load_dir',         dest='load_dir',         default='latest',
                           type=str, help='Directory to load old model,default "new" as the latest model')
    dir_parse.add_argument('--store_dir',        dest='store_dir',        default='latest',
                           type=str, help='Directory to store current model, default "latest" to save in timestamp')
    dir_parse.add_argument('--result_dir',       dest='result_dir',       default='result.csv',
                           type=str, help='Directory to store (history) performance result')
    dir_parse.add_argument('--builder_save_dir', dest='builder_save_dir', default='builder_save',
                           type=str, help='Directory to store current model for tfjs predictor')

    _args, _ = parser.parse_known_args()
    return vars(_args)


args = ParseArgs()
# Locate the data files
TF_files = find_training_data(
    args['start_date'], args['train_period'], os.getcwd() + args['train_data_path'])

tfrecordfiles = list(set(TF_files))


# Identify whether it's using NNI tuning mode
if args['tuning'] == 1:
    tuner_params = nni.get_next_parameter()
    try:
        args.update(tuner_params)
    except Exception as err:
        logging.error('Error args updated: %s', err)
        logging.error('Failed with params: %s', str(args))

DATASET_SIZE = 0
for fn in tfrecordfiles:
    for _ in tf.python_io.tf_record_iterator(fn):
        DATASET_SIZE += 1

training_size = int(DATASET_SIZE * args['train_ratio'])
args['num_training_min'] = int(
    training_size/args['batch_size']) * args['num_epochs_min']
args['num_training'] = int(
    training_size/args['batch_size']) * args['num_epochs']
args['validation_frequency'] = int((args['num_training_min'])/4)
if args['validation_frequency'] <= 0:
    args['validation_frequency'] = 1
args['print_train_iter'] = args['validation_frequency']
args['save_model'] = args['validation_frequency']

# 1: To one-hot-encode the features, 0: otherwise, use embeddings
if args['model'] == "LR":
    args['to_ohe'] = 1
elif args['model'] == "DNN":
    args['to_ohe'] = 0
else:
    logging.info("Check model input: Model is neither LR or DNN")

# HParams are used to group the hyper parameters
model_hps = tf.contrib.training.HParams(
    num_features=NUM_CATEGORICAL_FEATURES + 5,
    model=args['model'],
    loss=args['loss'],
    hidden_units=args['hidden_units'],
    learning_rate=args['learning_rate'],
    decay_step=args['decay_step'],
    decay_rate=args['decay_rate'],
    Lambda=args['Lambda'],
    gamma=args['gamma'],
    beta=args['beta'],
    drop_rate=args['drop_rate'],
    embedding_units=args['embedding_units'],
    embedding_units_ohe=args['embedding_units_ohe']
)

train_hps = tf.contrib.training.HParams(
    has_gpu=args['has_gpu'],
    is_test=args['is_test'],
    model=model_hps.model,
    num_training_min=args['num_training_min'],
    num_training=args['num_training'],
    print_train_iter=args['print_train_iter'],
    validation_frequency=args['validation_frequency'],
    validation_length=args['validation_length'],
    test_length=args['test_length'],
    save_model=args['save_model'],

    # early stopping
    earlystop_check_frequency=args['earlystop_check_frequency'],
    earlystop_duration=args['earlystop_duration'],
    valid_loss_delta=args['valid_loss_delta'],
    num_threshold_buffer=args['num_threshold_buffer'],
    percentile_threshold=args['percentile_threshold'],

    is_percentile_threshold=False
)

data_pipeline_hps = tf.contrib.training.HParams(
    has_gpu=train_hps.has_gpu,
    data_file=tfrecordfiles,
    is_test=train_hps.is_test,
    to_ohe=args['to_ohe'],
    random_seed=args['random_seed'],
    num_cores=args['num_cores'],
    train_ratio=args['train_ratio'],
    valid_ratio=args['valid_ratio'],
    batch_size=args['batch_size'],
    prefetch_size=args['prefetch_size'])

# Set model directory hyperparameters
args['model_name'] = args['model'] 
if args['store_dir'] == 'latest':
    store_dir = args['model_name'] + '_' + \
        parse_date('now').strftime("%Y-%m-%dT%H:%M:%S")
else:
    store_dir = args['store_dir']

if args['load_dir'] == 'latest':  # and new_run == 0
    load_dir = find_latest_model_dir(
        args['save_dir'], store_dir, args['model_name'])
else:
    load_dir = args['load_dir']

if args['new_run'] == 1:
    load_dir = store_dir

dir_hps = tf.contrib.training.HParams(
    save_dir=args['save_dir'],
    load_dir=load_dir,
    store_dir=store_dir,
    result_dir=args['save_dir'] + args['result_dir'],
    builder_save_dir=args['builder_save_dir']
)

Colnames = [dataset + metric for dataset in ['train_', 'valid_', 'test_']
            for metric in ['all_auc', 'batch_auc', 'all_apr', 'batch_apr', 'ctr', 'loss', 'prec', 'rec']]
