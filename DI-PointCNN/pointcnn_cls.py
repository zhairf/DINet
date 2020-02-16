from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pointfly as pf
import tensorflow as tf
from pointcnn import PointCNN


class Net(PointCNN):
    def __init__(self, points, features, is_training, setting):
        PointCNN.__init__(self, points, features, is_training, setting)
        fc_mean = tf.reduce_mean(self.fc_layers[-1], axis=1, keep_dims=True, name='fc_mean')
        
        self.feature_list = tf.reshape(self.feature_list,[128,61440])
        
        
        self.feature_list_A = self.feature_list[0:64]
        self.feature_list_B = self.feature_list[64:128]
        
        self.fc_layers[-1] = tf.cond(is_training, lambda: self.fc_layers[-1], lambda: fc_mean)#最后一层连接
       
        
        self.logits = pf.dense(self.fc_layers[-1], setting.num_class, 'logits',
                               is_training, with_bn=False, activation=None)