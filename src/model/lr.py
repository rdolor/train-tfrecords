import tensorflow as tf
import sys
sys.path.append('..')

from tensorflow.python.ops import array_ops

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name)


class LR(object):
    def __init__(self, model_hps):
        
        # Hyperparameters of the model
        self.learning_rate = model_hps.learning_rate
        self.decay_step = model_hps.decay_step
        self.decay_rate = model_hps.decay_rate
        self.Lambda = model_hps.Lambda
        self.gamma = model_hps.gamma      
        self.beta = model_hps.beta
        self.num_features = model_hps.num_features

    def core_builder(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.num_features), name='input_node')
        self.y = tf.placeholder(tf.float32, shape=(None, 1))
        self.loss_cond = tf.placeholder(tf.int16, shape=[])
        self.class_weights = tf.placeholder(dtype=tf.float32)
        self.phase = tf.placeholder(tf.bool, name='phase')
        self.drop_rate = tf.placeholder_with_default(1.0, shape=())

        self.w = weight_variable([self.num_features, 1], name='w')
        self.b = bias_variable([1], name='b')

        bn = tf.layers.batch_normalization(inputs=self.x, training=self.phase)
        logit = tf.matmul(bn, self.w) + self.b
        self.y_prob = tf.sigmoid(logit, name='prediction_node')


        # Selection of loss functions; set mse as 'default'
        mse = tf.reduce_mean(tf.square(tf.subtract(self.y, self.y_prob)))
        xloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logit))

        labels = tf.cast(self.y, dtype=tf.int32)
        weights = tf.gather(self.class_weights, labels)
        weighted_xloss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                                        targets=self.y, logits=logit,
                                        pos_weight=weights))

        # Regularizations
        l1_reg = tf.reduce_sum(tf.abs(self.w))
        l2_reg = tf.reduce_sum(tf.nn.l2_loss(self.w))

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
