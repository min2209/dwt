import tensorflow as tf

def angularErrorTotal(pred, gt, weight, ss, outputChannels=2):
    with tf.name_scope("angular_error"):
        pred = tf.reshape(pred, (-1, outputChannels))
        gt = tf.to_float(tf.reshape(gt, (-1, outputChannels)))
        weight = tf.to_float(tf.reshape(weight, (-1, 1)))
        ss = tf.to_float(tf.reshape(ss, (-1, 1)))

        pred = tf.nn.l2_normalize(pred, 1) * 0.999999
        gt = tf.nn.l2_normalize(gt, 1) * 0.999999

        errorAngles = tf.acos(tf.reduce_sum(pred * gt, reduction_indices=[1], keep_dims=True))

        lossAngleTotal = tf.reduce_sum((tf.abs(errorAngles*errorAngles))*ss*weight)

        return lossAngleTotal

def angularErrorLoss(pred, gt, weight, ss, outputChannels=2):
        lossAngleTotal = angularErrorTotal(pred=pred, gt=gt, ss=ss, weight=weight, outputChannels=outputChannels) \
                         / (countTotal(ss)+1)

        tf.add_to_collection('losses', lossAngleTotal)

        totalLoss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return totalLoss


def exceedingAngleThreshold(pred, gt, ss, threshold, outputChannels=2):
    with tf.name_scope("angular_error"):
        pred = tf.reshape(pred, (-1, outputChannels))
        gt = tf.to_float(tf.reshape(gt, (-1, outputChannels)))
        ss = tf.to_float(tf.reshape(ss, (-1, 1)))

        pred = tf.nn.l2_normalize(pred, 1) * 0.999999
        gt = tf.nn.l2_normalize(gt, 1) * 0.999999

        errorAngles = tf.acos(tf.reduce_sum(pred * gt, reduction_indices=[1], keep_dims=True)) * ss

        exceedCount = tf.reduce_sum(tf.to_float(tf.less(threshold/180*3.14159, errorAngles)))

        return exceedCount

def countCorrect(pred, gt, ss, k, outputChannels):
    with tf.name_scope("correct"):
        pred = tf.argmax(tf.reshape(pred, (-1, outputChannels)), 1)
        gt = tf.reshape(gt, (-1, outputChannels))

        ss = tf.to_float(tf.reshape(ss, (-1, 1)))

        correct = tf.reduce_sum(tf.mul(tf.reshape(tf.to_float(tf.nn.in_top_k(gt, pred, k)), (-1, 1)), ss), reduction_indices=[0])
        return correct


def countTotal(ss):
    with tf.name_scope("total"):
        ss = tf.to_float(tf.reshape(ss, (-1, 1)))
        total = tf.reduce_sum(ss)

        return total

def countTotalWeighted(ss, weight):
    with tf.name_scope("total"):
        ss = tf.to_float(tf.reshape(ss, (-1, 1)))
        weight = tf.to_float(tf.reshape(weight, (-1, 1)))
        total = tf.reduce_sum(ss * weight)

        return total