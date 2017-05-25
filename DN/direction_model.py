import numpy as np
from math import ceil
import tensorflow as tf
import scipy.io as sio

VGG_MEAN = [103.939, 116.779, 123.68]

class Network:
    def __init__(self, params, wd=5e-5, modelWeightPaths=None):
        self._params = params
        self._images = tf.placeholder("float")
        self._batch_images = tf.expand_dims(self._images, 0)
        self._gt = tf.placeholder("float")
        self._batch_gt = tf.expand_dims(self._gt, 0)
        self._wd = wd
        self.modelDict = {}

        if modelWeightPaths is not None:
            for path in modelWeightPaths:
                self.modelDict.update(sio.loadmat(path))

    def build(self, inputData, ss=None, ssMask=None, keepProb=1.0):
        # ss_3 = tf.tile(tf.expand_dims(ss, -1), [1, 1, 1, 3])
        inputData = inputData * tf.expand_dims(ss, -1)

        inputData = tf.concat(3, [inputData, tf.expand_dims(ssMask,-1)])

        self.conv1_1 = self._conv_layer(inputData, params=self._params["direction/conv1_1"])
        self.conv1_2 = self._conv_layer(self.conv1_1, params=self._params["direction/conv1_2"])
        self.pool1 = self._max_pool(self.conv1_2, 'direction/pool1')

        self.conv2_1 = self._conv_layer(self.pool1, params=self._params["direction/conv2_1"])
        self.conv2_2 = self._conv_layer(self.conv2_1, params=self._params["direction/conv2_2"])
        self.pool2 = self._max_pool(self.conv2_2, 'direction/pool2')

        self.conv3_1 = self._conv_layer(self.pool2, params=self._params["direction/conv3_1"])
        self.conv3_2 = self._conv_layer(self.conv3_1, params=self._params["direction/conv3_2"])
        self.conv3_3 = self._conv_layer(self.conv3_2, params=self._params["direction/conv3_3"])
        self.pool3 = self._average_pool(self.conv3_3, 'direction/pool3')

        self.conv4_1 = self._conv_layer(self.pool3, params=self._params["direction/conv4_1"])
        self.conv4_2 = self._conv_layer(self.conv4_1, params=self._params["direction/conv4_2"])
        self.conv4_3 = self._conv_layer(self.conv4_2, params=self._params["direction/conv4_3"])
        self.pool4 = self._average_pool(self.conv4_3, 'direction/pool4')

        self.conv5_1 = self._conv_layer(self.pool4, params=self._params["direction/conv5_1"])
        self.conv5_2 = self._conv_layer(self.conv5_1, params=self._params["direction/conv5_2"])
        self.conv5_3 = self._conv_layer(self.conv5_2, params=self._params["direction/conv5_3"])

        print "built all CNN layers!"

        self.fcn5_1 = self._conv_layer(self.conv5_3, params=self._params["direction/fcn5_1"])
        self.fcn5_2 = self._conv_layer(self.fcn5_1, params=self._params["direction/fcn5_2"])
        self.fcn5_3 = self._conv_layer(self.fcn5_2, params=self._params["direction/fcn5_3"])

        self.fcn4_1 = self._conv_layer(self.conv4_3, params=self._params["direction/fcn4_1"])
        self.fcn4_2 = self._conv_layer(self.fcn4_1, params=self._params["direction/fcn4_2"])
        self.fcn4_3 = self._conv_layer(self.fcn4_2, params=self._params["direction/fcn4_3"])

        self.fcn3_1 = self._conv_layer(self.conv3_3, params=self._params["direction/fcn3_1"])
        self.fcn3_2 = self._conv_layer(self.fcn3_1, params=self._params["direction/fcn3_2"])
        self.fcn3_3 = self._conv_layer(self.fcn3_2, params=self._params["direction/fcn3_3"])

        print "built all FCN layers!"

        self.upscore5_3 = self._upscore_layer(self.fcn5_3, params=self._params["direction/upscore5_3"],
                                           shape=tf.shape(self.fcn3_3))
        self.upscore4_3 = self._upscore_layer(self.fcn4_3, params=self._params["direction/upscore4_3"],
                                           shape=tf.shape(self.fcn3_3))

        self.fuse3 = tf.concat(3, [self.fcn3_3, self.upscore5_3, self.upscore4_3], name="direction/fuse3")
        self.fuse3_1 = self._conv_layer(self.fuse3, params=self._params["direction/fuse3_1"])
        self.fuse3_2 = self._conv_layer(self.fuse3_1, params=self._params["direction/fuse3_2"])
        self.fuse3_3 = self._conv_layer(self.fuse3_2, params=self._params["direction/fuse3_3"])
        print "built all fusing layers"

        self.output = self._upscore_layer(self.fuse3_3, params=self._params["direction/upscore3_1"],
                                          shape=tf.shape(inputData))

        # ss_2 = tf.tile(tf.expand_dims(ss,-1),[1,1,1,2])
        # self.output = self.output * ss_2
        self.output = self.output * tf.expand_dims(ss, -1)

        self.output = tf.nn.l2_normalize(self.output, 3, epsilon=1e-20)

        print "built the output layer!"
    # LAYER BUILDING

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _average_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _conv_layer(self, bottom, params, keepProb=1.0):
        with tf.variable_scope(params["name"]) as scope:
            filt = self.get_conv_filter(params)

            if "dr" in params.keys():
                conv = tf.nn.atrous_conv2d(bottom, filt, params["dr"], padding="SAME")
            else:
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(params)

            if params["act"] == "relu":
                activation = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
            elif params["act"] == "lin":
                activation = tf.nn.bias_add(conv, conv_biases)
            elif params["act"] == "tanh":
                activation = tf.nn.tanh(tf.nn.bias_add(conv, conv_biases))

            if not isinstance(keepProb, (int, long, float)):
                activation = tf.nn.dropout(activation, keep_prob=keepProb, seed=0)

        return activation

    # WEIGHTS GENERATION

    def get_bias(self, params):
        if params["name"]+"/biases" in self.modelDict:
            init = tf.constant_initializer(value=self.modelDict[params["name"]+"/biases"], dtype=tf.float32)
            print "loaded " + params["name"] + "/biases"
        else:
            init = tf.constant_initializer(value=0.0)
            print "generated " + params["name"] + "/biases"

        var = tf.get_variable(name="biases", initializer=init, shape=params["shape"][3])

        return var

    def get_conv_filter(self, params):
        if params["name"]+"/weights" in self.modelDict:
            init = tf.constant_initializer(value=self.modelDict[params["name"]+"/weights"], dtype=tf.float32)
            var = tf.get_variable(name="weights", initializer=init, shape=params["shape"])
            print "loaded " + params["name"]+"/weights"
        else:
            if params["std"]:
                stddev = params["std"]
            else:
                fanIn = params["shape"][0]*params["shape"][1]*params["shape"][2]
                stddev = (2/float(fanIn))**0.5

            init = tf.truncated_normal(shape=params["shape"], stddev=stddev, seed=0)
            var = tf.get_variable(name="weights", initializer=init)
            print "generated " + params["name"] + "/weights"

        if not tf.get_variable_scope().reuse:
            weightDecay = tf.mul(tf.nn.l2_loss(var), self._wd,
                                  name='weight_loss')
            tf.add_to_collection('losses', weightDecay)

        return var

    def _upscore_layer(self, bottom, shape, params):
        strides = [1, params["stride"], params["stride"], 1]
        with tf.variable_scope(params["name"]):
            in_features = bottom.get_shape()[3].value

            new_shape = [shape[0], shape[1], shape[2], params["outputChannels"]]
            output_shape = tf.pack(new_shape)

            f_shape = [params["ksize"], params["ksize"], params["outputChannels"], in_features]

            weights = self.get_deconv_filter(f_shape, params)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

        return deconv

    def get_deconv_filter(self, f_shape, params):
        if params["name"]+"/up_filter" in self.modelDict:
            init = tf.constant_initializer(value=self.modelDict[params["name"]+"/up_filter"], dtype=tf.float32)
        else:
            width = f_shape[0]
            height = f_shape[0]
            f = ceil(width / 2.0)
            c = (2 * f - 1 - f % 2) / (2.0 * f)
            bilinear = np.zeros([f_shape[0], f_shape[1]])
            for x in range(width):
                for y in range(height):
                    value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                    bilinear[x, y] = value
            weights = np.zeros(f_shape)
            for i in range(f_shape[2]):
                weights[:, :, i, i] = bilinear

            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)

        return tf.get_variable(name="up_filter", initializer=init, shape=f_shape)
