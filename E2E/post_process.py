import scipy.ndimage.interpolation
import scipy.misc
import skimage.morphology
import numpy as np

#CS codes: 24: person, 25: rider, 32: motorcycle, 33: bicycle, 26: car, 27: truck, 28: bus, 31: train
#PSP SS codes:
# CLASS_TO_SS = {"person":12, "rider":13, "motorcycle":18,
#                "bicycle":19, "car":13, "truck":15, "bus":16, "train":17}
CLASS_TO_SS = {"person":-128, "rider":-96, "motorcycle":-64,
               "bicycle":-32, "car":32, "truck":64, "bus":96, "train":128}
CLASS_TO_CITYSCAPES = {"person":24, "rider":25, "motorcycle":32,
               "bicycle":33, "car":26, "truck":27, "bus":28, "train":31}
THRESHOLD = {"person":1, "rider":1, "motorcycle":1, "bicycle":1,
             "car":2, "truck":2, "bus":2, "train":2}
MIN_SIZE = {"person":20, "rider":20, "motorcycle":20, "bicycle":20,
            "car":25, "truck":45, "bus":45, "train":45}
SELEM = {1: (np.ones((3,3))).astype(np.bool),
         2: (np.ones((5,5))).astype(np.bool)}

def watershed_cut(depthImage, ssMask):
    ssMask = ssMask.astype(np.int32)
    resultImage = np.zeros(shape=ssMask.shape, dtype=np.float32)

    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ssCode = CLASS_TO_SS[semClass]
        ssMaskClass = (ssMask == ssCode)

        ccImage = (depthImage > THRESHOLD[semClass]) * ssMaskClass
        ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE[semClass])
        ccImage = skimage.morphology.remove_small_holes(ccImage)
        ccLabels = skimage.morphology.label(ccImage)

        ccIDs = np.unique(ccLabels)[1:]
        for ccID in ccIDs:
            ccIDMask = (ccLabels == ccID)
            ccIDMask = skimage.morphology.binary_dilation(ccIDMask, SELEM[THRESHOLD[semClass]])
            instanceID = 1000 * csCode + ccID
            resultImage[ccIDMask] = instanceID

    resultImage = resultImage.astype(np.uint16)
    return resultImage









