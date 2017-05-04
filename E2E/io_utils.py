import numpy as np
import skimage
import skimage.io
import scipy.io as sio
import scipy.misc
import skimage.transform

np.random.seed(0)

VGG_MEAN = [103.939, 116.779, 123.68]
CLASS_TO_SS = {"person":11, "rider":12, "motorcycle":17,
               "bicycle":18, "car":13, "truck":14, "bus":15, "train":16}

def read_mat(path):
    return np.load(path)


def write_mat(path, m):
    np.save(path, m)

class Batch_Feeder:
    def __init__(self, dataset, train, batchSize, padWidth=None, padHeight=None, flip=False, keepEmpty=True, shuffle=False):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._dataset = dataset
        self._train = train
        self._batchSize = batchSize
        self._padWidth = padWidth
        self._padHeight = padHeight
        self._flip = flip
        self._keepEmpty = keepEmpty
        self._shuffle = shuffle

    def set_paths(self, idList=None, imageDir=None, gtDir=None, ssDir=None):
        self._paths = []

        if self._train:
            for id in idList:
                self._paths.append([id, imageDir + '/' + id + '_leftImg8bit.png',
                                    gtDir + '/' + id + '_unified_GT.mat',
                                    ssDir + '/' + id + '.png'])
            if self._shuffle:
                self.shuffle()
        else:
            for id in idList:
                self._paths.append([id, imageDir + '/' + id + '_leftImg8bit.png',
                                    ssDir + '/' + id + '.png'])

        self._numData = len(self._paths)

        if self._numData < self._batchSize:
            self._batchSize = self._numData

    def shuffle(self):
        np.random.shuffle(self._paths)

    def next_batch(self):
        idBatch = []
        imageBatch = []
        gtBatch = []
        ssBinaryBatch = []
        ssMaskBatch = []
        weightBatch = []

        if self._train:
            while(len(idBatch) < self._batchSize):
                ssImage = skimage.io.imread(self._paths[self._index_in_epoch][3])
                ssBinary, ssMask = ssProcess(ssImage)

                idBatch.append(self._paths[self._index_in_epoch][0])
                image = (image_scaling(skimage.io.imread(self._paths[self._index_in_epoch][1]))).astype(float)
                image = scipy.misc.imresize(image, 50)
                gt = (sio.loadmat(self._paths[self._index_in_epoch][2])['depth_map']).astype(float)
                weight = (sio.loadmat(self._paths[self._index_in_epoch][2])['weight_map']).astype(float)

                imageBatch.append(pad(image, self._padHeight, self._padWidth))
                gtBatch.append(pad(gt, self._padHeight, self._padWidth))
                weightBatch.append(pad(weight, self._padHeight, self._padWidth))
                ssBinaryBatch.append(pad(ssBinary, self._padHeight, self._padWidth))
                ssMaskBatch.append(pad(ssMask, self._padHeight, self._padWidth))

                self._index_in_epoch += 1

                if self._index_in_epoch == self._numData:
                    self._index_in_epoch = 0
                    if self._shuffle:
                        self.shuffle()

            imageBatch = np.array(imageBatch)
            gtBatch = np.array(gtBatch)
            ssBinaryBatch = np.array(ssBinaryBatch)
            ssMaskBatch = np.array(ssMaskBatch)
            weightBatch = np.array(weightBatch)

            if self._flip and np.random.uniform() > 0.5:
                for i in range(len(imageBatch)):
                    for j in range(3):
                        imageBatch[i,:,:,j] = np.fliplr(imageBatch[i,:,:,j])

                    ssBinaryBatch[i] = np.fliplr(ssBinaryBatch[i])
                    ssMaskBatch[i] = np.fliplr(ssMaskBatch[i])
                    gtBatch[i] = np.fliplr(gtBatch[i])
                    weightBatch[i] = np.fliplr(weightBatch[i])

            return imageBatch, gtBatch, weightBatch, ssBinaryBatch, ssMaskBatch, idBatch
        else:
            for example in self._paths[self._index_in_epoch:min(self._index_in_epoch+self._batchSize, self._numData)]:
                image = skimage.io.imread(example[1])
                image = scipy.misc.imresize(image,50)
                image = pad(image_scaling(image), self._padHeight, self._padWidth).astype(float)

                imageBatch.append(image)

                idBatch.append(example[0])
                ssImage = skimage.io.imread(example[2])

                ssImage = scipy.misc.imresize(ssImage, 50, interp="nearest")

                ssBinary, ssMask = ssProcess(ssImage)

                ssMaskBatch.append(pad(ssMask, self._padHeight, self._padWidth))
                ssBinaryBatch.append(pad(ssBinary, self._padHeight, self._padWidth))

            imageBatch = np.array(imageBatch)
            ssBinaryBatch = np.array(ssBinaryBatch)
            ssMaskBatch = np.array(ssMaskBatch)

            self._index_in_epoch += self._batchSize

            return imageBatch, ssBinaryBatch, ssMaskBatch, idBatch

    def total_samples(self):
        return self._numData

def read_ids(path):
    # return ['munster/munster_000071_000019']
    return [line.rstrip('\n') for line in open(path)]

def image_scaling(rgb_in):
    if rgb_in.dtype == np.float32:
        rgb_in = rgb_in*255
    elif rgb_in.dtype == np.uint8:
        rgb_in = rgb_in.astype(np.float32)

    # VGG16 was trained using opencv which reads images as BGR, but skimage reads images as RGB
    rgb_out = np.zeros(rgb_in.shape).astype(np.float32)
    rgb_out[:,:,0] = rgb_in[:,:,2] - VGG_MEAN[2]
    rgb_out[:,:,1] = rgb_in[:,:,1] - VGG_MEAN[1]
    rgb_out[:,:,2] = rgb_in[:,:,0] - VGG_MEAN[0]

    return rgb_out

def pad(data, padHeight=None, padWidth=None):
    if padHeight and padWidth:
        if data.ndim == 3:
            npad = ((0,padHeight-data.shape[0]),(0,padWidth-data.shape[1]),(0,0))
        elif data.ndim == 2:
            npad = ((0, padHeight - data.shape[0]), (0, padWidth - data.shape[1]))
        padData = np.pad(data, npad, mode='constant', constant_values=0)

    else:
        padData = data

    return padData

def ssProcess(ssImage):
    ssMask = np.zeros(shape=ssImage.shape, dtype=np.float32)
    ssImageInt = ssImage

    if ssImageInt.dtype == np.float32:
        ssImageInt = (ssImageInt*255).astype(np.uint8)

    # order: Person, Rider, Motorcycle, Bicycle, Car, Truck, Bus, Train

    ssMask += (ssImageInt==CLASS_TO_SS['person']).astype(np.float32)*1
    ssMask += (ssImageInt==CLASS_TO_SS['rider']).astype(np.float32)*2
    ssMask += (ssImageInt==CLASS_TO_SS['motorcycle']).astype(np.float32)*3
    ssMask += (ssImageInt==CLASS_TO_SS['bicycle']).astype(np.float32)*4
    ssMask += (ssImageInt==CLASS_TO_SS['car']).astype(np.float32)*6
    ssMask += (ssImageInt==CLASS_TO_SS['truck']).astype(np.float32)*7
    ssMask += (ssImageInt==CLASS_TO_SS['bus']).astype(np.float32)*8
    ssMask += (ssImageInt==CLASS_TO_SS['train']).astype(np.float32)*9

    ssBinary = (ssMask != 0).astype(np.float32)

    ssMask[ssMask == 0] = 1 # temp fix

    ssMask = (ssMask - 5) * 32

    return ssBinary, ssMask



