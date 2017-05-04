import math
import tensorflow as tf
import numpy as np
import sys
import os
import scipy.io as sio
import skimage.io

VGG_MEAN = [103.939, 116.779, 123.68]

tf.set_random_seed(0)

def initialize_model(outputChannels, wd=None, modelWeightPaths=None):
    params = {
        "direction/conv1_1": {"name": "direction/conv1_1", "shape": [3, 3, 3, 64], "std": None, "act": "relu"},
        "direction/conv1_2": {"name": "direction/conv1_2", "shape": [3, 3, 64, 64], "std": None, "act": "relu"},
        "direction/conv2_1": {"name": "direction/conv2_1", "shape": [3, 3, 64, 128], "std": None, "act": "relu"},
        "direction/conv2_2": {"name": "direction/conv2_2", "shape": [3, 3, 128, 128], "std": None, "act": "relu"},
        "direction/conv3_1": {"name": "direction/conv3_1", "shape": [3, 3, 128, 256], "std": None, "act": "relu"},
        "direction/conv3_2": {"name": "direction/conv3_2", "shape": [3, 3, 256, 256], "std": None, "act": "relu"},
        "direction/conv3_3": {"name": "direction/conv3_3", "shape": [3, 3, 256, 256], "std": None, "act": "relu"},
        "direction/conv4_1": {"name": "direction/conv4_1", "shape": [3, 3, 256, 512], "std": None, "act": "relu"},
        "direction/conv4_2": {"name": "direction/conv4_2", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/conv4_3": {"name": "direction/conv4_3", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/conv5_1": {"name": "direction/conv5_1", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/conv5_2": {"name": "direction/conv5_2", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/conv5_3": {"name": "direction/conv5_3", "shape": [3, 3, 512, 512], "std": None, "act": "relu"},
        "direction/fcn5_1": {"name": "direction/fcn5_1", "shape": [5, 5, 512, 512], "std": 1e-4, "act": "relu"},
        "direction/fcn5_2": {"name": "direction/fcn5_2", "shape": [1, 1, 512, 512], "std": 1e-4, "act": "relu"},
        "direction/fcn5_3": {"name": "direction/fcn5_3", "shape": [1, 1, 512, 256], "std": 1e-4, "act": "relu"},
        "direction/upscore5_3": {"name": "direction/upscore5_4", "ksize": 8, "stride": 4, "outputChannels": 256},
        "direction/fcn4_1": {"name": "direction/fcn4_1", "shape": [5, 5, 512, 512], "std": 1e-4, "act": "relu"},
        "direction/fcn4_2": {"name": "direction/fcn4_2", "shape": [1, 1, 512, 512], "std": 1e-4, "act": "relu"},
        "direction/fcn4_3": {"name": "direction/fcn4_3", "shape": [1, 1, 512, 256], "std": 1e-4, "act": "relu"},
        "direction/upscore4_3": {"name": "direction/upscore4_3", "ksize": 4, "stride": 2, "outputChannels": 256},
        "direction/fcn3_1": {"name": "direction/fcn3_1", "shape": [5, 5, 256, 256], "std": 1e-5, "act": "relu"},
        "direction/fcn3_2": {"name": "direction/fcn3_2", "shape": [1, 1, 256, 256], "std": 1e-5, "act": "relu"},
        "direction/fcn3_3": {"name": "direction/fcn3_3", "shape": [1, 1, 256, 256], "std": 1e-5, "act": "relu"},
        "direction/fuse3_1": {"name": "direction/fuse_1", "shape": [1,1,256*3,512], "std": 1e-5, "act": "relu"},
        "direction/fuse3_2": {"name": "direction/fuse_2", "shape": [1, 1, 512, 512], "std": 1e-5, "act": "relu"},
        "direction/fuse3_3": {"name": "direction/fuse_3", "shape": [1, 1, 512, 2], "std": 1e-5, "act": "lin"},
        "direction/upscore3_1": {"name": "direction/upscore3_1", "ksize": 8, "stride": 4, "outputChannels": 2},

        "depth/conv1_1": {"name": "depth/conv1_1", "shape": [5,5,2,64], "std": 1e-1, "act": "relu"},
        "depth/conv1_2": {"name": "depth/conv1_2", "shape": [5,5,64,128], "std": 1e-1, "act": "relu"},
        "depth/conv2_1": {"name": "depth/conv2_1", "shape": [5,5,128,128], "std": 1e-2, "act": "relu"},
        "depth/conv2_2": {"name": "depth/conv2_2", "shape": [5,5,128,128], "std": 1e-2, "act": "relu"},
        "depth/conv2_3": {"name": "depth/conv2_3", "shape": [5,5,128,128], "std": 1e-2, "act": "relu"},
        "depth/conv2_4": {"name": "depth/conv2_4", "shape": [5,5,128,128], "std": 1e-2, "act": "relu"},
        "depth/fcn1": {"name": "depth/fcn1", "shape": [1,1,128,128], "std": 1e-2, "act": "relu"},
        "depth/fcn2": {"name": "depth/fcn2", "shape": [1,1,128,16], "std": 1e-1, "act": "relu"},
        "depth/upscore": {"name": "depth/upscore", "ksize": 8, "stride": 4, "outputChannels": 16},
        }

    return joint_model2_wide.Network(params, wd=wd, modelWeightPaths=modelWeightPaths)

def forward_model(model, feeder, outputSavePath):
    with tf.Session() as sess:
        images = tf.placeholder("float")
        tfBatchImages = tf.expand_dims(images, 0)
        ss = tf.placeholder("float")
        tfBatchSS = tf.expand_dims(ss, 0)
        keepProb = tf.placeholder("float")

        with tf.name_scope("model_builder"):
            print "attempting to build model"
            model.build(tfBatchImages, tfBatchSS, keepProb=keepProb)
            print "built the model"

        init = tf.initialize_all_variables()

        sess.run(init)

        if not os.path.exists(outputSavePath):
            os.makedirs(outputSavePath)
        # for i in range(1):
        for i in range(int(math.floor(feeder.total_samples() / batchSize))):
            imageBatch, ssBatch, idBatch = feeder.next_batch()
            # skimage.io.imsave("/u/mbai/transfer/scaledimage.png",imageBatch[0,:,:,:])
            # sio.savemat("/u/mbai/transfer/scaledimage.mat",{'image':imageBatch[0,:,:,:]})
            # raw_input("saved")

            outputBatch = sess.run(model.outputDataArgMax, feed_dict={tfBatchImages: imageBatch,
                                                                      tfBatchSS: ssBatch,
                                                                      keepProb: 1.0})
            outputBatch = outputBatch.astype(np.uint8)

            # outputBatch = sess.run(model.direction, feed_dict={tfBatchImages: imageBatch,
            #                                                           tfBatchSS: ssBatch,
            #                                                           keepProb: 1.0})

            for j in range(len(idBatch)):
                outputFilePath = os.path.join(outputSavePath, idBatch[j]+'.mat')
                # outputFilePath = os.path.join(outputSavePath, idBatch[j] + '.png')
                outputFileDir = os.path.dirname(outputFilePath)

                if not os.path.exists(outputFileDir):
                    os.makedirs(outputFileDir)

                sio.savemat(outputFilePath, {"depth_map": outputBatch[j]}, do_compression=True)

                # skimage.io.imsave(outputFilePath, outputBatch[j])

                # sio.savemat(outputFilePath, {"dir_map": outputBatch[j]})

            print "processed image %d to %d out of %d"%(i*batchSize+1, (i+1)*batchSize, feeder.total_samples())
            sys.stdout.flush()

if __name__ == "__main__":
    outputChannels = 16
    outputPrefix = "submission3"
    outputSet = 'val'
    batchSize = 10

    configurations = {'car': {"index":[0], "model": ["/ais/gobi4/mbai/instance_seg/cityscapes/models/joint/joint2_vehicles_final_wideup_ssLRR_045.mat"]},
                      'truck': {"index":[5], "model": ["/ais/gobi4/mbai/instance_seg/cityscapes/models/joint/joint2_vehicles_final_wideup_ssLRR_045.mat"]},
                      'bus': {"index": [6], "model": ["/ais/gobi4/mbai/instance_seg/cityscapes/models/joint/joint2_vehicles_final_wideup_ssLRR_045.mat"]},
                      'train': {"index": [7], "model": ["/ais/gobi4/mbai/instance_seg/cityscapes/models/joint/joint2_vehicles_final_wideup_ssLRR_045.mat"]},
                      'person': {"index": [1], "model": ["/ais/gobi4/mbai/instance_seg/cityscapes/models/joint/joint2_humans_final_wideup_ssLRR_010.mat"]},
                      'rider': {"index": [2], "model": ["/ais/gobi4/mbai/instance_seg/cityscapes/models/joint/joint2_humans_final_wideup_ssLRR_010.mat"]},
                      'motorcycle': {"index": [3], "model": ["/ais/gobi4/mbai/instance_seg/cityscapes/models/joint/joint2_cycles_final_wideup_ssLRR_010.mat"]},
                      'bicycle': {"index": [4], "model": ["/ais/gobi4/mbai/instance_seg/cityscapes/models/joint/joint2_cycles_final_wideup_ssLRR_010.mat"]},
                      }
    # 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train

    for type in configurations:
        model = initialize_model(outputChannels=outputChannels, modelWeightPaths=configurations[type]["model"])

        feeder = Batch_Feeder(dataset="cityscapes",
                                      indices=configurations[type]["index"],
                                      train=False,
                                      batchSize=batchSize)

        feeder.set_paths(idList=read_ids('/ais/gobi4/mbai/instance_seg/cityscapes/splits/'+outputSet+'list.txt'),
                         imageDir="/ais/gobi4/mbai/instance_seg/cityscapes/inputImages/"+outputSet,
                            ssDir="/ais/gobi4/mbai/instance_seg/cityscapes/unified/ssMaskFineLRR/"+outputSet)

        forward_model(model, feeder=feeder,
                      outputSavePath="/ais/gobi4/mbai/instance_seg/training/outputs/%s/%s/%s"%(outputPrefix, outputSet, type))

        tf.reset_default_graph()
