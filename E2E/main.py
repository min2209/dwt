from network_init import get_model
from io_utils import *
import tensorflow as tf
from forward import forward_model
from train import train_model

tf.set_random_seed(0)

if __name__ == "__main__":
    outputChannels = 16
    savePrefix = ""
    outputPrefix = ""
    # 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
    train = False

    if train:
        batchSize = 3
        learningRate = 5e-6 # usually i use 5e-6
        wd = 1e-6

        modelWeightPaths = [""]

        initialIteration = 1

        trainFeeder = Batch_Feeder(dataset="cityscapes",
                                           train=train,
                                           batchSize=batchSize,
                                           flip=True, keepEmpty=False, shuffle=True)

        trainFeeder.set_paths(idList=read_ids(''),
                         imageDir="n",
                         gtDir="",
                         ssDir="")

        valFeeder = Batch_Feeder(dataset="cityscapes",
                                         train=train,
                                         batchSize=batchSize, shuffle=False)

        valFeeder.set_paths(idList=read_ids(''),
                         imageDir="",
                         gtDir="",
                         ssDir="")

        model = get_model(wd=wd, modelWeightPaths=modelWeightPaths)

        train_model(model=model, outputChannels=outputChannels,
                    learningRate=learningRate,
                    trainFeeder=trainFeeder,
                    valFeeder=valFeeder,
                    modelSavePath="",
                    savePrefix=savePrefix,
                    initialIteration=initialIteration)

    else:
        batchSize = 1
        modelWeightPaths = ["../model/dwt_cityscapes_pspnet.mat"]

        model = get_model(modelWeightPaths=modelWeightPaths)

        feeder = Batch_Feeder(dataset="cityscapes",
                                      train=train,
                                      batchSize=batchSize)

        feeder.set_paths(idList=read_ids('../example/sample_list.txt'),
                         imageDir="../example/inputImages",
                            ssDir="../example/PSPNet")

        forward_model(model, feeder=feeder,
                      outputSavePath="../example/output")
