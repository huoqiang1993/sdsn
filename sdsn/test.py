import numpy as np
import scipy.misc
from PIL import Image
import scipy.io
import os
import scipy
import sys

caffe_root = '..'
sys.path.insert(0, caffe_root+'/python/')
import caffe

database = 'DRIVE'

# Use GPU?
use_gpu = 1;
gpu_id = 0;

net_struct  = 'deploy_dsn.prototxt'
data_source = './data/test_DRIVE.txt'

# Input your path here
data_root = ''
save_root = './results/'+database+'/'

if not os.path.exists(save_root):
    os.makedirs(save_root)
    
with open(data_source) as f:
    imnames = f.readlines()

test_lst = [data_root + x.strip() for x in imnames]

if use_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)


# load net
net = caffe.Net('./'+net_struct,'./pretrained_sdsn.caffemodel', caffe.TEST);
	
for idx in range(0,len(test_lst)):
    print("Scoring sdsn for image " + data_root + imnames[idx][:-1])
    
    #Read and preprocess data
    im = Image.open(test_lst[idx])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1] #BGR
    in_ -= np.array((171.0773,98.4333,58.8811)) #Mean substraction
    in_ = in_.transpose((2,0,1))
    
    #Reshape data layer
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    
    #Score the model
    net.forward()
    fuse  = net.blobs['sigmoid-fuse'].data[0][0,:,:]
   
   #Save the results
    scipy.misc.imsave(save_root+imnames[idx][:-1], fuse)
