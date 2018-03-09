#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidFineCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const SigmoidFineCEParameter  fine_ce_param = this->layer_param_.sigmoid_fine_ce_loss_param();
  range_ = fine_ce_param.threshold_range();
  alpha_ = fine_ce_param.threshold_alpha();
  beta_  = fine_ce_param.threshold_beta();

  CHECK_GT(range_, 1) << "Number of neighboring pixels to take into account must be greater thatn 1.";
  LOG(INFO) << "This implementation does not support train batch size > 1.";
}

template <typename Dtype>
void SigmoidFineCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_FINE_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidFineCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  //const int count = bottom[0]->count();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;
  Dtype *weight_pos; // the weight vector of positive pixels 
  Dtype *weight_neg; // the weight vector of negative pixels
  int wpi = 0;       // points to wpi-th element of vector weight_pos
  int wni = 0;       // points to wni-th element of vector weight_neg

//  Dtype wv_min, wv_max;
  Dtype wp_sum = 0;
  Dtype wn_sum = 0;
  Dtype wn_max;

  // compute the number of positive and negative points
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int j = 0; j < dim; j++) {
	if (target[j] == 1) {
	  count_pos++;
	} else if (target[j] == 0) {
	  count_neg++;
	} else {
      LOG(FATAL) << "Target must be either 0 or 1.";
    }
  }

  weight_pos = (Dtype *)malloc(sizeof(Dtype)*count_pos);
  weight_neg = (Dtype *)malloc(sizeof(Dtype)*count_neg);

  // compute weight_pos
  for (int i = 0; i < height; i++) {
	for (int j = 0; j < width; j++) {
	  if (target[i*width+j] == 1) {
	    //compute the weight score for wpi-th element
	    if ((i-(range_-1)/2) < 0 || (j-(range_-1)/2) < 0 ||
	      (i+(range_-1)/2) >= height || (j+(range_-1)/2) >= width) {
		  weight_pos[wpi++] = 1;
	    } else {
		  int temp_count_pos = 0;
		  for (int row = i-(range_-1)/2; row <= i+(range_-1)/2; row++) {
			for (int col = j-(range_-1)/2; col <= j+(range_-1)/2; col++) {
			  if (target[row*width+col] == 1) {
			    temp_count_pos++;
			  }
			}
		  }
		  weight_pos[wpi++] = temp_count_pos;
		}
	  }
	}
  }
  CHECK_EQ(wpi, count_pos) << "weight vector has not been filled up yet.";
  //Normalize weight_pos
  for (int i = 0; i < count_pos; i++) {
	if (weight_pos[i] <= 4) weight_pos[i] = 3;
	else weight_pos[i] = 1;
//	weight_pos[i] =  range_ * range_ - weight_pos[i]+1;
	wp_sum += weight_pos[i];
  }
  for (int i = 0; i < count_pos; i++) {
	weight_pos[i] *= count_pos/wp_sum;
  }

  // compute weight_neg
  for (int i = 0; i < height; i++) {
	for (int j = 0; j < width; j++) {
	  if (target[i*width+j] == 0) { //negative pixels
		int found = 0;
		for (int k = 0;; k++) {
		  if ((i-k) >= 0 && (j-k) >= 0 && (j+k) < width) { // up -----
			for (int m = j-k; m <= j+k; m++) {
			  if (target[(i-k)*width+m] == 1) { found = 1; break; }
			}
		  }
		  if ((i-k) >= 0 && (j-k) >= 0 && (i+k) < height) { //left down
			for (int m = i-k; m <= i+k; m++) {
			  if (target[(m*width+(j-k))] == 1) {found = 1; break;}
			}
		  }
		  if ((i+k) < height && (j-k) >= 0 && (j+k) < width) {//down -----
			for (int m = j-k; m <= j+k; m++) {
			  if (target[(i+k)*width+m] == 1) { found = 1; break; }
			}
		  }
		  if ((i-k) >= 0 && (i+k) < height && (j+k) < width) { //right down
			for (int m = i-k; m <= i+k; m++) {
			  if (target[(m*width+(j+k))] == 1) {found = 1; break;}
			}
		  }
		  if (((i-k)*width+(j-k) > 0 && ((i-k)*width+(j-k) < height*width)) && target[(i-k)*width+(j-k)] == 1) {found = 1; weight_neg[wni++] = k; break;}
		  if (((i+k)*width+(j+k) > 0 && ((i+k)*width+(j+k) < height*width)) && target[(i+k)*width+(j+k)] == 1) {found = 1; weight_neg[wni++] = k; break;}
		  if (((i+k)*width+(j-k) > 0 && ((i+k)*width+(j-k) < height*width)) && target[(i+k)*width+(j-k)] == 1) {found = 1; weight_neg[wni++] = k; break;}
		  if (((i-k)*width+(j+k) > 0 && ((i-k)*width+(j+k) < height*width)) && target[(i-k)*width+(j+k)] == 1) {found = 1; weight_neg[wni++] = k; break;}
		  if (found == 1) {
		    weight_neg[wni++] = k;
		    break;
		  }
		}
	  }
    }
  }
  // Normalize weight_neg
  CHECK_GT(count_neg, 0);
  wn_max = weight_neg[0];
  for (int i = 1; i < count_neg; i++) {
	wn_max = (weight_neg[i] > wn_max) ? weight_neg[i] : wn_max; 
  }
  for (int i = 0; i < count_neg; i++) {
	weight_neg[i] = wn_max - weight_neg[i] + 1;
    wn_sum += weight_neg[i];
  }
  for (int i = 0; i < count_neg; i++) {
	weight_neg[i] *= count_neg/wn_sum;
  }
  
#if 0
  //Normalize weight vector to [alpha, beta]
  CHECK_GT(count_pos, 0);
  wv_max = wv_min = weight_pos[0];
  for (int i = 1; i < count_pos; i++) {
	  wv_max = (weight_pos[i] > wv_max) ? weight_pos[i] : wv_max; 
	  wv_min = (weight_pos[i] < wv_min) ? weight_pos[i] : wv_min; 
  }
  for (int i = 0; i < count_pos; i++) {
	  weight_pos[i] = alpha_ + (beta_ - alpha_)/(wv_max-wv_min)*(weight_pos[i]-wv_min);
  }
#endif

  // compute the loss value
  wpi = 0;
  wni = 0;
  for (int j = 0; j < dim; j++) {
	if (target[j] == 1) {
     loss_pos -= weight_pos[wpi++]*(input_data[j]*(target[j]-(input_data[j]>=0)) - log(1 + exp(input_data[j] - 2*input_data[j]*(input_data[j] >= 0))));
//       loss_pos -= (input_data[j]*(target[j]-(input_data[j]>=0)) - log(1 + exp(input_data[j] - 2*input_data[j]*(input_data[j] >= 0))));
	} else if (target[j] == 0) {
//	   loss_neg -= weight_neg[wni++]*(input_data[j]*(target[j] - (input_data[j]>=0)) - log(1 + exp(input_data[j] - 2*input_data[j]*(input_data[j] >= 0))));
	  loss_neg -= (input_data[j]*(target[j] - (input_data[j]>=0)) - log(1 + exp(input_data[j] - 2*input_data[j]*(input_data[j] >= 0))));
	} else {
      LOG(FATAL) << "Target must be either 0 or 1.";
	}
  }
  loss_pos *= count_neg / (count_pos + count_neg);
  loss_neg *= count_pos / (count_pos + count_neg);
  top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg);
  free(weight_pos);
  free(weight_neg);
}

template <typename Dtype>
void SigmoidFineCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
	const int height = bottom[0]->height();
	const int width  = bottom[0]->width();
    Dtype count_pos = 0;
    Dtype count_neg = 0;
	Dtype *weight_pos;
	Dtype *weight_neg;
//	Dtype wv_min, wv_max;
	int wpi = 0, wni = 0;
	Dtype wp_sum = 0, wn_sum = 0;
	Dtype wn_max;

    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);

  // compute the number of positive and negative points
    int dim = bottom[0]->count() / bottom[0]->num();
    for (int j = 0; j < dim; j++) {
      if (target[j] == 1) {
  	    count_pos++;
  	  } else if (target[j] == 0) {
  	    count_neg++;
  	  } else {
        LOG(FATAL) << "Target must be either 0 or 1.";
      }
    }

    weight_pos = (Dtype *)malloc(sizeof(Dtype)*count_pos);
    weight_neg = (Dtype *)malloc(sizeof(Dtype)*count_neg);

  // compute weight_pos
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        if (target[i*width+j] == 1) {
          //compute the weight score for wpi-th element
          if ((i-(range_-1)/2) < 0 || (j-(range_-1)/2) < 0 ||
            (i+(range_-1)/2) >= height || (j+(range_-1)/2) >= width) {
      	      weight_pos[wpi++] = 1;
          } else {
      	     int temp_count_pos = 0;
      	     for (int row = i-(range_-1)/2;row <= i+(range_-1)/2; row++) {
      		  for (int col = j-(range_-1)/2;col <= j+(range_-1)/2;col++) {
      			if (target[row*width+col] == 1) {
      			  temp_count_pos++;
      			}
      		  }
      	    }
      	     weight_pos[wpi++] = temp_count_pos;
      	  }
        }
      }
    }
    CHECK_EQ(wpi, count_pos) << "weight vector has not been filled up yet.";
    for (int i = 0; i < count_pos; i++) {
   //   weight_pos[i] =  range_ * range_ - weight_pos[i]+1;
	  if (weight_pos[i] <= 4) weight_pos[i] = 3;
	  else weight_pos[i] = 1;
	  wp_sum += weight_pos[i];
    }
    for (int i = 0; i < count_pos; i++) {
	  weight_pos[i] *= count_pos/wp_sum;
    }
#if 0
    //Normalize weight vector to [alpha, beta]
    CHECK_GT(count_pos, 0);
    wv_max = wv_min = weight_pos[0];
    for (int i = 1; i < count_pos; i++) {
        wv_max = (weight_pos[i] > wv_max) ? weight_pos[i] : wv_max; 
        wv_min = (weight_pos[i] < wv_min) ? weight_pos[i] : wv_min; 
    }
    for (int i = 0; i < count_pos; i++) {
        weight_pos[i] = alpha_ + (beta_ - alpha_)/(wv_max-wv_min)*(weight_pos[i]-wv_min);
    }
#endif
    // compute weight_neg
  for (int i = 0; i < height; i++) {
	for (int j = 0; j < width; j++) {
	  if (target[i*width+j] == 0) { //negative pixels
		int found = 0;
		for (int k = 0;; k++) {
		  if ((i-k) >= 0 && (j-k) >= 0 && (j+k) < width) { // up -----
			for (int m = j-k; m <= j+k; m++) {
			  if (target[(i-k)*width+m] == 1) { found = 1; break; }
			}
		  }
		  if ((i-k) >= 0 && (j-k) >= 0 && (i+k) < height) { //left down
			for (int m = i-k; m <= i+k; m++) {
			  if (target[(m*width+(j-k))] == 1) {found = 1; break;}
			}
		  }
		  if ((i+k) < height && (j-k) >= 0 && (j+k) < width) {//down -----
			for (int m = j-k; m <= j+k; m++) {
			  if (target[(i+k)*width+m] == 1) { found = 1; break; }
			}
		  }
		  if ((i-k) >= 0 && (i+k) < height && (j+k) < width) { //right down
			for (int m = i-k; m <= i+k; m++) {
			  if (target[(m*width+(j+k))] == 1) {found = 1; break;}
			}
		  }
		  if (((i-k)*width+(j-k) > 0 && ((i-k)*width+(j-k) < height*width)) && target[(i-k)*width+(j-k)] == 1) {found = 1; weight_neg[wni++] = k; break;}
		  if (((i+k)*width+(j+k) > 0 && ((i+k)*width+(j+k) < height*width)) && target[(i+k)*width+(j+k)] == 1) {found = 1; weight_neg[wni++] = k; break;}
		  if (((i+k)*width+(j-k) > 0 && ((i+k)*width+(j-k) < height*width)) && target[(i+k)*width+(j-k)] == 1) {found = 1; weight_neg[wni++] = k; break;}
		  if (((i-k)*width+(j+k) > 0 && ((i-k)*width+(j+k) < height*width)) && target[(i-k)*width+(j+k)] == 1) {found = 1; weight_neg[wni++] = k; break;}
		  if (found == 1) {
		    weight_neg[wni++] = k;
		    break;
		  }
		}
	  }
    }
  }
    // Normalize weight_neg
    CHECK_GT(count_neg, 0);
    wn_max = weight_neg[0];
    for (int i = 1; i < count_neg; i++) {
      wn_max = (weight_neg[i] > wn_max) ? weight_neg[i] : wn_max; 
    }
    for (int i = 0; i < count_neg; i++) {
      weight_neg[i] = wn_max - weight_neg[i] + 1;
      wn_sum += weight_neg[i];
    }
    for (int i = 0; i < count_neg; i++) {
      weight_neg[i] *= count_neg/wn_sum;
    }
	/* compute gradient */
	wpi = 0;
	wni = 0;
	for (int j = 0; j < dim; j++) {
	  if (target[j] == 0) {
		bottom_diff[j] *= count_pos / (count_pos + count_neg);
//		bottom_diff[j] *= weight_neg[wni++];
	  } else if (target[j] == 1){
	    bottom_diff[j] *= count_neg / (count_pos + count_neg);
	    bottom_diff[j] *= weight_pos[wpi++];
	  } else {
	    LOG(FATAL) << "Target must be either 0 or 1.";
	  }
	}
	free(weight_pos);
	free(weight_neg);

    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidFineCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidFineCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidFineCrossEntropyLoss);

}  // namespace caffe
