## Deep Watershed Transform 

Performs instance level segmentation detailed in the following paper: 

Min Bai and Raquel Urtasun, Deep Watershed Transformation for Instance Segmentation, in CVPR 2017. Accessible at https://arxiv.org/abs/1611.08303. 

This page is still under construction. 

## Dependencies

Developed and tested on Ubuntu 14.04 and 16.04. 

1) TensorFlow www.tensorflow.org 
2) Numpy, Scipy, and Skimage (sudo apt-get install python-numpy python-scipy python-skimage)

## Inputs

1) Cityscapes images (www.cityscapes-dataset.com). 
2) Semantic Segmentation for input images. In our case, we used the output from PSPNet (by H. Zhao et al. https://github.com/hszhao/PSPNet). These are uint8 images with pixel-wise semantic labels encoded with 'trainIDs' defined by Cityscapes. For more information, visit https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

## Outputs

The model produces pixel-wise instance labels as a uint16 image with the same formatting as the Cityscapes instance segmentation challenge ground truth. In particular, each pixel is labeled as 'id' * 1000 + instance_id, where 'id' is as defined by Cityscapes (for more information, consult labels.py in the above link), and instance_id is an integer indexing the object instance. 

## Testing the Model

1) Clone repository into dwt/.
2) Download the model from www.cs.toronto.edu/~mbai/dwt_cityscapes_pspnet.mat and place into the "dwt/model" directory.
3) run "cd E2E"
4) run "python main.py"
5) The results will be available in "dwt/example/output".

## Training the Model

1) Will be available soon. 
