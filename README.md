## Deeply supervised neural network with short connections for retinal vessel segmentation
Please read our paper (https://arxiv.org/abs/1803.03963) for more details!

### Introduction:
In this paper, we present a multi-scale and multi-level deeply supervised convolutional neural network with short connections (SDSN) for vessel segmentation. We use short connections to transfer semantic information between side-output layers. Forward short connections could pass low level semantic information to high level and backward short connections could pass much structural information to low level. In addition, we propose using a structural similarity measurement to evaluate the vessel map. The proposed method was verified on DRIVE dataset and it showed superior performance compared with other state-of-the-art methods. Specially, with patch level input, the network gets 0.7890 sensitivity, 0.9803 specificity and 0.9802 AUC.

# Training SDSN
1. Download the DRIVE dataset from (https://www.isi.uu.nl/Research/Databases/DRIVE/download.php).
2. Download fully convolutional VGG model (248MB) from (http://vcl.ucsd.edu/hed/5stage-vgg.caffemodel) and put it in $CAFFE_ROOT/sdsn/.	
3. Build Caffe
4. Run the python scripts in $CAFFE_ROOT/sdsn
	```bash
	python train.py
	```
# Testing SDSN
1. Clone the respository
	```bash
	git clone https://github.com/guomugong/sdsn.git
	```
2. Build Caffe
	```bash
	cp Makefile.config.example Makefile.config
	make all -j8
	make pycaffe
	```
3. Prepare your retinal images and set its path in test.py
4. Run
	```bash
	python test.py
	```

# Acknowledgment
	This code is based on HED. Thanks to their contributions.
	@inproceedings{xie2015holistically,
	  title={Holistically-nested edge detection},
      author={Xie, Saining and Tu, Zhuowen},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      pages={1395--1403},
      year={2015}
    }
## License
This code can not be used for commercial applications
