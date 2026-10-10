#include <cuda_runtime.h>
#include <iostream>
#include "globals.h"
#include "random_weights_matrix.h"

using namespace std;

__global__ void feature_compute(const float* d_features, const float* d_weights, float* d_next_features, const int* d_indices, const int* d_indptr, const int num_nodes, const int num_features) 
{
    int feature_idx=blockIdx.x*blockDim.x+threadIdx.x;

    
    if (feature_idx >= num_features*num_nodes) return;

    extern __shared__ float s_vsum[];
    
    int node_idx = blockIdx.x;


    s_vsum[threadIdx.x] = d_features[feature_idx];
    
    // vado a prendere feature vicini e aggrego
    for (int i=d_indptr[node_idx]; i<d_indptr[node_idx+1]; i++)
    {
      
        s_vsum[threadIdx.x]+= d_features[d_indices[i]*num_features+threadIdx.x];
        
    }

    // apsetto che finisca aggregazione feature
    __syncthreads();

    // faccio update della feature
    float sum=0.0f;
    for (int k=0; k<num_features; k++)
        {
            sum+= d_weights[threadIdx.x*num_features+k]*s_vsum[k];
        }

        d_next_features[feature_idx]=fmaxf(0.0f, sum);
    
    
}

__global__ void logit_compute_final(const float* d_features, const float* d_weights_out, float* d_logits, const int num_nodes, const int num_features, const int num_classes) 
{
    int logit_idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (logit_idx >= num_nodes*num_classes) return;

    int node_idx=blockIdx.x;

    float sum=0.0f;
    for (int k=0; k<num_features; k++)
    {
        sum+= d_weights_out[k*num_classes+threadIdx.x]*d_features[node_idx*num_features+k];
    }
    d_logits[logit_idx]=sum;

}

void inference_engine_cuda_thread_per_feature() 
{
    float* h_features=features.data();
    float* h_weights_out=weights_out.data();
    float* h_next_features=next_features.data();
    float* h_logits=logits.data();
    int* h_indices=indices.data();
    int* h_indptr=indptr.data();

    int num_nodes=meta["num_nodes"];
    int num_features=meta["num_features"];
    int num_classes=meta["num_classes"];

    float* d_features; 
    float* d_weights_out; 
    float* d_next_features; 
    float* d_logits;
    int* d_indices; 
    int* d_indptr;



    cudaMalloc((void**)&d_features, num_nodes*num_features*sizeof(float));
    cudaMalloc((void**)&d_weights_out, num_features*num_classes*sizeof(float));
    cudaMalloc((void**)&d_next_features, num_nodes*num_features*sizeof(float));
    cudaMalloc((void**)&d_logits, num_nodes*num_classes*sizeof(float));
    cudaMalloc((void**)&d_indices, indptr.back()*sizeof(int));
    cudaMalloc((void**)&d_indptr, (num_nodes+1)*sizeof(int));
    

    cudaMemcpy(d_features, h_features, num_nodes*num_features*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_out, h_weights_out, num_features*num_classes*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, indptr.back()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, h_indptr, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);

    size_t shared_bytes = num_features * sizeof(float);

    int num_blocks = num_nodes;
    dim3 dim_grid(num_blocks, 1, 1);
    dim3 dim_block(num_features, 1, 1);
    dim3 dim_block_final(num_classes, 1, 1);

    for (int i = 0; i<NUM_LAYERS; i++)
    {
        if (i < NUM_LAYERS-1) 
        {
            const vector<float>& weights = layer_weights[i];
            const float* h_weights=weights.data();
            float* d_weights;
            cudaMalloc((void**)&d_weights, num_features*num_features*sizeof(float));
            cudaMemcpy(d_weights, h_weights, num_features*num_features*sizeof(float), cudaMemcpyHostToDevice);
            
            feature_compute<<<dim_grid, dim_block, shared_bytes>>>(d_features, d_weights, d_next_features, d_indices, d_indptr, num_nodes, num_features);

            cudaDeviceSynchronize();
            
            float* temp = d_features;
            d_features = d_next_features;
            d_next_features = temp;

            cudaFree(d_weights);

        }
        else 
        {
            // lancio un thread per logit
            logit_compute_final<<<dim_grid, dim_block_final>>>(d_features, d_weights_out, d_logits, num_nodes, num_features, num_classes);

            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(logits.data(), d_logits, num_nodes*num_classes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_features);
    cudaFree(d_weights_out);
    cudaFree(d_next_features);
    cudaFree(d_logits);
    cudaFree(d_indices);
    cudaFree(d_indptr);
    

}
