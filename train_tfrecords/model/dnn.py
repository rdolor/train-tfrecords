import tensorflow as tf
import sys
sys.path.append('..')

from train_tfrecords.config import NUM_WEEKDAY,NUM_REGION,NUM_CITY,NUM_ADEXCHANGE,NUM_SLOTFORMAT,NUM_USERTAG
    
from tensorflow.python.ops import array_ops

class DNN(object):
    def __init__(self, model_hps):
        
        # Hyperparameters of the model
        self.hidden_units = model_hps.hidden_units
        self.embedding_units = model_hps.embedding_units
        self.embedding_units_ohe = model_hps.embedding_units_ohe
        self.learning_rate = model_hps.learning_rate
        self.decay_step = model_hps.decay_step
        self.decay_rate = model_hps.decay_rate
        self.Lambda = model_hps.Lambda
        self.gamma = model_hps.gamma
        self.beta = model_hps.beta                     

    def core_builder(self):

        # categorical features to use embedding
        features_to_embed = ['weekday', 'region','city', 'adexchange', 'slotformat']
        features_num_item = [NUM_WEEKDAY,NUM_REGION,NUM_CITY,NUM_ADEXCHANGE,NUM_SLOTFORMAT]
            
        self.features_dict = {} # FORMAT:{key(feature_name):value([# items, # embedding units, placeholder, weight, embedding lookup weight])}
        for feature_name, num_item, num_embed_units in zip(features_to_embed,features_num_item,self.embedding_units):
            if feature_name not in self.features_dict.keys():
                self.features_dict[feature_name] = []
            self.features_dict[feature_name].extend([num_item, num_embed_units])

        for f_name in self.features_dict.keys():
            self.features_dict[f_name].append(tf.placeholder(tf.int32, shape=(None, 1),name=f_name))
            self.features_dict[f_name].append(tf.get_variable(f_name + "_emb_w", [self.features_dict[f_name][0], self.features_dict[f_name][1]],
                                    initializer=tf.random_normal_initializer(), dtype=tf.float32))
            self.features_dict[f_name].append(tf.nn.embedding_lookup(self.features_dict[f_name][3], self.features_dict[f_name][2]))
        
        # Numeric input
        self.hour = tf.placeholder(tf.float32, shape=(None,1),name="hour")
        self.slotwidth = tf.placeholder(tf.float32, shape=(None,1),name="slotwidth")
        self.slotheight = tf.placeholder(tf.float32, shape=(None,1),name="slotheight")
        self.slotvisibility = tf.placeholder(tf.float32, shape=(None,1),name="slotvisibility") 
        self.slotprice = tf.placeholder(tf.float32, shape=(None,1),name="slotprice")

        # Array input
        self.usertag = tf.placeholder(tf.float32, shape=(None, NUM_USERTAG),name="usertag")
        self.usertag_dense = tf.layers.dense(self.usertag, self.embedding_units_ohe[0], activation=tf.nn.elu)

        self.x = tf.concat([self.usertag_dense,
                            self.hour, self.slotwidth, self.slotheight, self.slotvisibility,self.slotprice], 1)
        
        for f_name in self.features_dict.keys():
            self.x = tf.concat([self.x, tf.reshape(self.features_dict[f_name][4], [-1, self.features_dict[f_name][1]])], axis=1)

        self.y = tf.placeholder(tf.float32, shape=(None, 1))
        self.loss_cond = tf.placeholder(tf.int16, shape=[])
        self.class_weights = tf.placeholder(dtype=tf.float32)
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.drop_rate = tf.placeholder_with_default(1.0, shape=())
        
        # Model
        bn = tf.layers.batch_normalization(inputs=self.x, training=self.phase)
        dnn = tf.layers.dense(bn, self.hidden_units[0], activation=tf.nn.elu, name='dnn1')
        dnn = tf.layers.dropout(dnn, self.drop_rate)

        for hidden_unit_idx in range(1,len(self.hidden_units)):
            dnn = tf.layers.dense(dnn, self.hidden_units[hidden_unit_idx], activation=tf.nn.elu, name='dnn'+str(hidden_unit_idx+1))
            dnn = tf.layers.dropout(dnn, self.drop_rate)

        logit = tf.reshape(tf.layers.dense(dnn, 1, activation=None, name='output_layer'), [-1, 1])
        self.y_prob = tf.nn.sigmoid(logit, 'prediction_node')

        # Selection of loss functions; set mse as 'default'
        mse = tf.reduce_mean(tf.square(tf.subtract(self.y, self.y_prob)))
        xloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logit))

        labels = tf.cast(self.y, dtype=tf.int32)
        weights = tf.gather(self.class_weights, labels)
        weighted_xloss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                                        targets=self.y, logits=logit,
                                        pos_weight=weights))

        # Regularizations
        l1_reg = 0
        for f_name in self.features_dict.keys():
            l1_reg += tf.reduce_sum(tf.abs(self.features_dict[f_name][3]))

        l2_reg = 0
        for f_name in self.features_dict.keys():
            l2_reg += tf.nn.l2_loss(self.features_dict[f_name][3])

        # Focal loss (Ref: Focal Loss for Dense Object Detection => cost-sensitive loss for imbalanced data)
        # FL(p_t) = − α_t * (1 − p_t )^γ  * log(p_t)
        # In general α should be decreased slightly as γ is increased; γ = 2, α = 0.25 works best
        # alpha => self.Lambda => [0.,1.]
        # gamma => self.gamma  => [0.,5.], 2 usually works well

        labels = tf.to_float(labels)
        zeros = array_ops.zeros_like(self.y_prob, dtype=self.y_prob.dtype)
        pos_p_sub = array_ops.where(labels > zeros, labels - self.y_prob, zeros)
        neg_p_sub = array_ops.where(labels > zeros, zeros, self.y_prob)
        per_entry_cross_ent = - self.Lambda * (pos_p_sub ** self.gamma) * tf.log(tf.clip_by_value(self.y_prob, 1e-8, 1.0))-(1-self.Lambda) * (neg_p_sub ** self.gamma) * tf.log(tf.clip_by_value(1.0 - self.y_prob, 1e-8, 1.0))
        focal_loss = tf.reduce_sum(per_entry_cross_ent)

        self.loss_op = tf.cond(tf.equal(self.loss_cond,10), lambda: mse                               , lambda: mse) #default
        self.loss_op = tf.cond(tf.equal(self.loss_cond,11), lambda: mse           + self.Lambda*l1_reg, lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,12), lambda: mse           + self.Lambda*l2_reg, lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,20), lambda: xloss                             , lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,21), lambda: xloss         + self.Lambda*l1_reg, lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,22), lambda: xloss         + self.Lambda*l2_reg, lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,30), lambda: weighted_xloss                    , lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,31), lambda: weighted_xloss+ self.Lambda*l1_reg, lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,32), lambda: weighted_xloss+ self.Lambda*l2_reg, lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,40), lambda: focal_loss                        , lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,41), lambda: focal_loss    + self.beta*l1_reg  , lambda: mse)
        self.loss_op = tf.cond(tf.equal(self.loss_cond,42), lambda: focal_loss    + self.beta*l2_reg  , lambda: mse)


        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   global_step,
                                                   self.decay_step,
                                                   self.decay_rate,
                                                   staircase=False)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(self.loss_op, global_step=global_step)