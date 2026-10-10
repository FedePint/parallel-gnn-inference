#include <cuda_runtime.h>
#include <iostream>
#include "globals.h"
#include "random_weights_matrix.h"

using namespace std;

const int BLOCK_SIZE=256;   

__global__ void edge_agg(const float* d_features, const float* d_weights, float* d_next_features, const int* d_edge_dest, float* d_vsum, const int* d_indices, const int* d_indptr, const int num_nodes, const int num_features, const int num_edges) 
{
    int edge_idx=blockIdx.x*blockDim.x+threadIdx.x;

    if (edge_idx >= num_edges) return;

    //mi cerco source e dest
    int dest = d_edge_dest[edge_idx];
    int source = d_indices[edge_idx];
    
    //aggiungo features
    for (int i=0; i<num_features; i++)
    {
        atomicAdd(&d_vsum[dest*num_features+i], d_features[source*num_features+i]);
    }

}
    
    

__global__ void feature_update(const float* d_features, const float* d_weights, float* d_next_features, const int* d_edge_dest,const float* d_vsum, const int* d_indices, const int* d_indptr, const int num_nodes, const int num_features, const int num_edges)
{
    int feature_idx=blockIdx.x*blockDim.x+threadIdx.x;
    if (feature_idx >= num_nodes*num_features) return;


    

        float sum=0.0f;
        for (int k=0; k<num_features; k++)
        {
            sum+= d_weights[threadIdx.x*num_features+k]*d_vsum[blockIdx.x*num_features+k];
        }

        d_next_features[feature_idx]=fmaxf(0.0f, sum);
    
}

__global__ void logit_compute_final_edge(const float* d_features, const float* d_weights_out, float* d_logits, const int num_nodes, const int num_features, const int num_classes) 
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

void inference_engine_cuda_thread_per_edge() 
{
    int* h_edge_dest=edge_dest.data();
    float* h_features=features.data();
    float* h_weights_out=weights_out.data();
    float* h_next_features=next_features.data();
    float* h_logits=logits.data();
    int* h_indices=indices.data();
    int* h_indptr=indptr.data();
    float* d_vsum;

    int num_nodes=meta["num_nodes"];
    int num_features=meta["num_features"];
    int num_classes=meta["num_classes"];
    int num_edges=meta["num_edges"];

    float* d_features; 
    float* d_weights_out; 
    float* d_next_features; 
    float* d_logits;
    int* d_indices; 
    int* d_indptr;
    int* d_edge_dest;



    cudaMalloc((void**)&d_features, num_nodes*num_features*sizeof(float));
    cudaMalloc((void**)&d_weights_out, num_features*num_classes*sizeof(float));
    cudaMalloc((void**)&d_next_features, num_nodes*num_features*sizeof(float));
    cudaMalloc((void**)&d_logits, num_nodes*num_classes*sizeof(float));
    cudaMalloc((void**)&d_indices, indptr.back()*sizeof(int));
    cudaMalloc((void**)&d_indptr, (num_nodes+1)*sizeof(int));
    cudaMalloc((void**)&d_edge_dest, edge_dest.size()*sizeof(int));
    cudaMalloc((void**)&d_vsum, num_nodes*num_features*sizeof(float));
    

    cudaMemcpy(d_features, h_features, num_nodes*num_features*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_out, h_weights_out, num_features*num_classes*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, indptr.back()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indptr, h_indptr, (num_nodes+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_dest, h_edge_dest, edge_dest.size()*sizeof(int), cudaMemcpyHostToDevice);
    
    
    

    int num_blocks = (meta["num_edges"]+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 dim_grid(num_blocks, 1, 1);
    dim3 dim_block(BLOCK_SIZE, 1, 1);
    dim3 dim_block_final(num_classes, 1, 1);
    dim3 dim_grid_update(num_nodes, 1, 1);
    dim3 dim_block_update(num_features, 1, 1);


    for (int i = 0; i<NUM_LAYERS; i++)
    {
        
        if (i < NUM_LAYERS-1) 
        {
            cudaMemcpy(d_vsum, d_features, num_nodes*num_features*sizeof(float), cudaMemcpyDeviceToDevice);
            const vector<float>& weights = layer_weights[i];
            const float* h_weights=weights.data();
            float* d_weights;
            cudaMalloc((void**)&d_weights, num_features*num_features*sizeof(float));
            cudaMemcpy(d_weights, h_weights, num_features*num_features*sizeof(float), cudaMemcpyHostToDevice);
            
            edge_agg<<<dim_grid, dim_block>>>(d_features, d_weights, d_next_features, d_edge_dest, d_vsum, d_indices, d_indptr, num_nodes, num_features, num_edges);

            // aspetto che finisca aggregazione feature
            cudaDeviceSynchronize();

            
            // aggiorno feature usando somma e matrix
            feature_update<<<dim_grid_update, dim_block_update>>>(d_features, d_weights, d_next_features, d_edge_dest, d_vsum, d_indices, d_indptr, num_nodes, num_features, num_edges);

            cudaDeviceSynchronize();
            
            float* temp = d_features;
            d_features = d_next_features;
            d_next_features = temp;

            cudaFree(d_weights);

        }
        else 
        {

            logit_compute_final_edge<<<dim_grid_update, dim_block_final>>>(d_features, d_weights_out, d_logits, num_nodes, num_features, num_classes);

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
