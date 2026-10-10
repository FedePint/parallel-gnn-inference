#include <cuda_runtime.h>
#include <iostream>
#include "globals.h"
#include "random_weights_matrix.h"

using namespace std;
const int DIM_BLOCK=256;

__global__ void node_compute(const float* d_features, const float* d_weights, float* d_next_features, float* d_vsum, const int* d_indices, const int* d_indptr, const int num_nodes, const int num_features) 
{
    int node_idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (node_idx >= num_nodes) return;
    
    //self loop
    for (int i=0; i<num_features; i++)
    {
        d_vsum[node_idx*num_features+i] = d_features[node_idx*num_features+i];
    }

    //faccio aggrgazione vicini
    for (int i=d_indptr[node_idx]; i<d_indptr[node_idx+1]; i++)
    {
        for (int k=0; k<num_features; k++)
        {
            d_vsum[node_idx*num_features+k]+= d_features[d_indices[i]*num_features+k];
        }
    }


    //update
    for(int i=0; i<num_features; i++)
    {
        float sum=0.0f;
        for (int k=0; k<num_features; k++)
        {
            sum+= d_weights[i*num_features+k]*d_vsum[node_idx*num_features+k];
        }

        d_next_features[node_idx*num_features+i]=fmaxf(0.0f, sum);
    }
}

__global__ void node_compute_final(const float* d_features, const float* d_weights_out, float* d_logits, const int* d_indices, const int* d_indptr, const int num_nodes, const int num_features, const int num_classes) 
{
    int node_idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (node_idx >= num_nodes) return;

    for (int i=0; i<num_classes; i++)
    {
        float sum=0.0f;
        for (int k=0; k<num_features; k++)
        {
            sum+= d_weights_out[k*num_classes+i]*d_features[node_idx*num_features+k];
        }
        d_logits[node_idx*num_classes+i]=sum;
    }




    

}

void inference_engine_cuda_node() 
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
    float* d_vsum;
    int* d_indices; 
    int* d_indptr;

    cudaMalloc((void**)&d_features, num_nodes*num_features*sizeof(float));
    cudaMalloc((void**)&d_weights_out, num_features*num_classes*sizeof(float));
    cudaMalloc((void**)&d_next_features, num_nodes*num_features*sizeof(float));
    cudaMalloc((void**)&d_logits, num_nodes*num_classes*sizeof(float));
    cudaMalloc((void**)&d_indices, indptr.back()*sizeof(int));
    cudaMalloc((void**)&d_indptr, (num_nodes+1)*sizeof(int));
    cudaMalloc((void**)&d_vsum, num_nodes*num_features*sizeof(float));

    cudaMemcpy(d_features, h_features, num_nodes*num_features*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_out, h_weights_out, num_features*num_classes*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, indptr.back()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, h_indptr, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vsum, h_features, num_nodes*num_features*sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks = (num_nodes+DIM_BLOCK-1)/DIM_BLOCK;
    dim3 dim_grid(num_blocks, 1, 1);
    dim3 dim_block(DIM_BLOCK, 1, 1);

    for (int i = 0; i<NUM_LAYERS; i++)
    {
        if (i < NUM_LAYERS-1) 
        {
            const vector<float>& weights = layer_weights[i];
            const float* h_weights=weights.data();
            float* d_weights;
            cudaMalloc((void**)&d_weights, num_features*num_features*sizeof(float));
            cudaMemcpy(d_weights, h_weights, num_features*num_features*sizeof(float), cudaMemcpyHostToDevice);
            
            node_compute<<<dim_grid, dim_block>>>(d_features, d_weights, d_next_features, d_vsum, d_indices, d_indptr, num_nodes, num_features);

            // sincronizzo per aggiornare feature per prossimo layer
            cudaDeviceSynchronize();
            
            float* temp = d_features;
            d_features = d_next_features;
            d_next_features = temp;

            cudaFree(d_weights);

        }
        else 
        {
            node_compute_final<<<dim_grid, dim_block>>>(d_features, d_weights_out, d_logits, d_indices, d_indptr, num_nodes, num_features, num_classes);

            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(logits.data(), d_logits, num_nodes*num_classes*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_features);
    cudaFree(d_weights_out);
    cudaFree(d_next_features);
    cudaFree(d_logits);
    cudaFree(d_vsum);
    cudaFree(d_indices);
    cudaFree(d_indptr);
    

}
