## S-DSN: Deeply Supervised Neural Network with Short Connections for Retinal Vessel Segmentation
Please read our [paper] (https://arxiv.org/abs/1803.03963) for more details!

### Introduction:
The condition of vessel of the human eye is an important factor for the diagnosis of ophthalmological diseases. Vessel segmentation in fundus images is a challenging task due to complex vessel structure, the presence of similar structures such as microaneurysms and hemorrhages, micro-vessel with only one to several pixels wide, and requirements for finer results. In this paper, we present a multi-scale deeply supervised network with short connections (S-DSN) for vessel segmentation.  We used short connections to transfer semantic information between side-output layers. Forward short connections pass low level semantic information to high level for refining results in high-level side-outputs, and backward short connection passes much structural information to low level for reducing noises in low-level side-outputs. In addition to traditional pixel-wise evaluation criteria for semantic segmentation, we introduced s-score to evaluate the structure similarity between segmentation results and ground truth, which is more in line with human feelings. The proposed S-DSN has been verified on DRIVE and STARE datasets, and showed superior performance than other state-of-the-art methods. Specially, with patch level input, the network achieved 0.7922/0.8132 sensitivity, 0.9799/0.9843 specificity, 0.9805/0.9850 AUC, 0.7842/0.8363 s-score, and 0.8244/0.8375 F1-score on DRIVE and STARE, respectively.
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
