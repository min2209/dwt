import tensorflow as tf

def depthCELoss2(pred, gt, weight, ss, outputChannels=16):
    with tf.name_scope("depth_CE_loss"):
        pred = tf.reshape(pred, (-1, outputChannels))
        epsilon = tf.constant(value=1e-25)
        predSoftmax = tf.to_float(tf.nn.softmax(pred))

        gt = tf.one_hot(indices=tf.to_int32(tf.squeeze(tf.reshape(gt, (-1, 1)))), depth=outputChannels, dtype=tf.float32)
        ss = tf.to_float(tf.reshape(ss, (-1, 1)))
        weight = tf.to_float(tf.reshape(weight, (-1, 1)))

        crossEntropyScaling = tf.to_float([3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        crossEntropy = -tf.reduce_sum(((1-gt)*tf.log(tf.maximum(1-predSoftmax, epsilon))
                                       + gt*tf.log(tf.maximum(predSoftmax, epsilon)))*ss*crossEntropyScaling*weight,
                                      reduction_indices=[1])

        crossEntropySum = tf.reduce_sum(crossEntropy, name="cross_entropy_sum")
        return crossEntropySum

def depthCELoss(pred, gt, ss, outputChannels=16):
    with tf.name_scope("depth_CE_loss"):
        pred = tf.reshape(pred, (-1, outputChannels))
        epsilon = tf.constant(value=1e-25)
        #pred = pred + epsilon
        predSoftmax = tf.to_float(tf.nn.softmax(pred))
        predSoftmax = predSoftmax + epsilon

        gt = tf.one_hot(indices=tf.to_int32(tf.squeeze(tf.reshape(gt, (-1, 1)))), depth=outputChannels, dtype=tf.float32)
        ss = tf.to_float(tf.reshape(ss, (-1, 1)))

        crossEntropy = -tf.reduce_sum(gt * tf.log(predSoftmax) * ss, reduction_indices=[1])

        crossEntropySum = tf.reduce_sum(crossEntropy, name="cross_entropy_sum")
        return crossEntropySum

def modelTotalLoss(pred, gt, weight, ss, outputChannels=1):
    lossDepthTotal = depthCELoss2(pred=pred, gt=gt, weight=weight, ss=ss,
                                  outputChannels=outputChannels) / (countTotalWeighted(ss, weight) + 1)

    tf.add_to_collection('losses', lossDepthTotal)

    totalLoss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return totalLoss

def countTotal(ss):
    with tf.name_scope("total"):
        ss = tf.to_float(tf.reshape(ss, (-1, 1)))
        total = tf.reduce_sum(ss)

        return total

def countCorrect(pred, gt, ss, k, outputChannels):
    with tf.name_scope("correct"):
        pred = tf.argmax(tf.reshape(pred, (-1, outputChannels)), 1)
        gt = tf.one_hot(indices=tf.to_int32(tf.squeeze(tf.reshape(gt, (-1, 1)))), depth=outputChannels, dtype=tf.float32)

        ss = tf.to_float(tf.reshape(ss, (-1, 1)))

        correct = tf.reduce_sum(tf.mul(tf.reshape(tf.to_float(tf.nn.in_top_k(gt, pred, k)), (-1, 1)), ss), reduction_indices=[0])
        return correct

def countTotalWeighted(ss, weight):
    with tf.name_scope("total"):
        ss = tf.to_float(tf.reshape(ss, (-1, 1)))
        weight = tf.to_float(tf.reshape(weight, (-1, 1)))
        total = tf.reduce_sum(ss * weight)

        return total