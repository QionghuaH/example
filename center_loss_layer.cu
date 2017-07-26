#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Compute_pos_distance_data_gpu(int nthreads, const int K, const Dtype* bottom,
	      const Dtype* label, const Dtype* center, Dtype* pos_distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    // distance(i) = x(i) - c_{y(i)}
    pos_distance[index] = bottom[index] - center[label_value * K + k];
  }
}

template <typename Dtype>
__global__ void Compute_pos_distance_val_data_gpu(int nthreads,  const int K /*feature len*/, const Dtype* pos_distance,
        Dtype* pos_distance_val) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //idx = M
    pos_distance_val[index] = 0.;
    for (int k = 0; k < K; k++) {
      pos_distance_val[index] += pos_distance[index * K + k] * pos_distance[index * K + k];
    }
  }
}

template <typename Dtype>
__global__ void Compute_neg_distance_data_val_gpu(int nthreads, const int N, const int M, const int K, const Dtype* bottom,
        const Dtype* center, const Dtype* pos_distance_val, Dtype* neg_distance_val, Dtype* mn_idxs, Dtype* nm_idxs) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // idx = M_ * N_ 
    int m = index / N;
    int n = index % N;


    neg_distance_val[index] = 0.;
    mn_idxs[m * N + n] = 0.;
    nm_idxs[n * M + m] = 0.;
    Dtype sum_val(0.);
    // d_{mn} = ||x_{m} - c_{n} ||^2 
    for (int k = 0; k < K; k++) {
      Dtype x = (bottom[m * K + k] - center[n * K + k]);
      sum_val  += (x * x);
    }
    //pos_dist[m] - d_{mn}
    if (sum_val < pos_distance_val[m]){
      neg_distance_val[index] = pos_distance_val[m] - sum_val;
      mn_idxs[m * N + n] = 1.;
      nm_idxs[n * M + m] = 1.;
    }
  }
}

 


template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*> & bottom,
    const vector<Blob<Dtype>*>& top) {
  int nthreads = M_ * K_;
  Compute_pos_distance_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                                this->blobs_[0]->gpu_data(), pos_distance_.mutable_gpu_data());
  Dtype pos_dot;
  caffe_gpu_dot(M_ * K_, pos_distance_.gpu_data(), pos_distance_.gpu_data(), &pos_dot);
  Dtype pos_loss = pos_dot / M_ / Dtype(2);

  // compute the pos dist val
  nthreads = M_;
  Compute_pos_distance_val_data_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, pos_distance_.gpu_data(), pos_distance_val_.mutable_gpu_data());

  // compute the neg dist val
  nthreads = M_ * N_;
  Compute_neg_distance_data_val_gpu<<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, M_, K_, bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), 
        pos_distance_val_.gpu_data(), neg_distance_val_.mutable_gpu_data(), mn_idxs_.mutable_gpu_data(), nm_idxs_.mutable_gpu_data());


  Dtype neg_loss(0.);
  Dtype count(0);
  caffe_gpu_asum(M_ * N_, neg_distance_val_.gpu_data(), &neg_loss);
  caffe_gpu_asum(M_ * N_, nm_idxs_.gpu_data(), &count);

  neg_loss = neg_loss / (count + (Dtype)1.) / M_;

  top[0]->mutable_cpu_data()[0] = pos_loss + neg_loss * gamma;
}



/**********************************************************************************************/
template <typename Dtype>
__global__ void Compute_pos_center_diff_gpu(int nthreads, const int M, const int N, const int K, const Dtype gamma,
        const Dtype* label, const Dtype* pos_distance, const Dtype* nm_idxs, Dtype* variation_sum, 
        Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    //c_{m}
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index) {
        count++;
        //找出是否存在 neg_dist < pos_dist 的情况
        int acc = 1;
        for (int n = 0; n < N; n++) {
          if (nm_idxs[n * M + m] > 0.1 && (static_cast<int>(label[m]) != index)) {
            acc = acc + gamma;
            break;
          }
        }

        for (int k = 0; k < K; k++) {
          variation_sum[index * K + k] -= (pos_distance[m * K + k] * acc);
        }
      }
    }
    
    

    for (int k = 0; k < K; k++) {
      center_diff[index * K + k] = variation_sum[index * K + k]  / (count + (Dtype)1.);
    }
  }
}


template <typename Dtype>
__global__ void Compute_neg_center_diff_gpu(int nthreads, const int M, const int K, const Dtype* label,
        const Dtype* bottom, const Dtype* center, const Dtype* nm_idxs, Dtype* variation_sum) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //index = N_
    int count = 0;

    for (int m = 0; m < M; m++) {

      if (nm_idxs[index * M + m] > 0.1 && (static_cast<int>(label[m]) != index)) {
        count++;
        for (int k = 0; k < K; k++) {
          //x_{m} - c_{n}
          variation_sum[index * K + k] += (bottom[m * K + k] - center[index * K + k]);
        }
        
      }
    }

    for (int k = 0; k < K; k++) {
      variation_sum[index * K + k] /=  (count + (Dtype)1.);
    }

  }
}


template <typename Dtype>
__global__ void Compute_X_diff_gpu(int nthreads, const int N, const int M, const int K, const Dtype* label,
        const Dtype* center, const Dtype* nm_idxs, Dtype* neg_distance) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    //idx = M_

     for (int n = 0; n < N; n++) {
      if (nm_idxs[n * M + index] > 0.1) {
        const int label_value = static_cast<int>(label[index]);
        count++;
        for (int k = 0; k < K; k++) {
          //c_{m} - c_{n}
          neg_distance[index * K + k] += (center[n * K + k] - center[label_value * K + k]);
        }
        
      }
    }

    for (int k = 0; k < K; k++) {
      neg_distance[index * K + k] = neg_distance[index * K + k]  / (count + (Dtype)1.);
    }
  }
}


template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {


    //使用c_{m} - x{m} 更新参数   
    int nthreads = N_;
    caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_gpu_data());
    Compute_pos_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, N_, K_, gamma, bottom[1]->gpu_data(), pos_distance_.gpu_data(),  nm_idxs_.gpu_data(),
                                  variation_sum_.mutable_gpu_data(), this->blobs_[0]->mutable_gpu_diff());


    nthreads = N_;
    caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_gpu_data());
    Compute_neg_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, bottom[1]->gpu_data(), bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), 
        nm_idxs_.gpu_data(), variation_sum_.mutable_gpu_data());

    //更新参数
    caffe_gpu_axpy(N_ * K_, gamma, variation_sum_.gpu_data(), this->blobs_[0]->mutable_gpu_diff());



    if (propagate_down[0]) {
      caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
                              pos_distance_.gpu_data(), bottom[0]->mutable_gpu_diff());

      nthreads = M_;
      caffe_gpu_set(M_ * K_, (Dtype)0., neg_distance_.mutable_gpu_data());
      Compute_X_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, N_, M_, K_, bottom[1]->gpu_data(), this->blobs_[0]->gpu_data(), nm_idxs_.gpu_data(),
          neg_distance_.mutable_gpu_data());

      caffe_gpu_axpy(M_ * K_, top[0]->cpu_diff()[0] / M_ * gamma,
                              neg_distance_.gpu_data(), bottom[0]->mutable_gpu_diff());
    }
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
                << " Layer cannot backpropagate to label inputs.";
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);

}  // namespace caffe
