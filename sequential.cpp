#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>
#include "bin_conversion.h"
#include "random_weights_matrix.h"

using namespace std;

#include "globals.h"

void inference_engine_sequential(const vector<float>& features_in, const vector<int>& indices, const vector<int>& indptr)
{
  vector<float> features = features_in;
  for (int i=0; i<NUM_LAYERS; i++)
  {
    cout << "[Checkpoint] Avvio Layer " << i+1 << "..." << endl;
    auto start_layer = chrono::high_resolution_clock::now();
    
    next_features.clear();
    const vector<float>& weights = layer_weights[i];
    if (i<NUM_LAYERS-1) 
    {
        int num_features = meta["num_features"];
        int num_nodes = meta["num_nodes"];
        next_features.resize(num_nodes * num_features);
        vector<float> vsum(num_features); 
        
        for (int j=0; j<num_nodes; j++)
        {
            //  svuoto e fillo il vettore somma
            fill(vsum.begin(), vsum.end(), 0.0f);
            
            
            for (int k=0; k<num_features; k++) 
            {
                vsum[k] += features[j*num_features + k];
            }
            
            // ciclo sui vicini
            for (int c=indptr[j]; c<indptr[j+1]; c++)
            {
                for (int k=0; k<num_features; k++) 
                {
                    vsum[k] += features[indices[c]*num_features + k];
                }
            }
            
            // W*h e RELU
            for (int r = 0; r<num_features; r++) 
            {
                float sum = 0.0f;
                for (int c=0; c<num_features; c++)
                {
                    sum += weights[r*num_features+c] * vsum[c];
                }
                next_features[j*num_features + r] = max(0.0f, sum);
            }
        }
        features = next_features;
    }
    else 
    {
        int num_features = meta["num_features"];
        int num_nodes = meta["num_nodes"];
        int num_classes = meta["num_classes"];
        for (int j=0; j<num_nodes; j++)
        {
            for (int c=0; c<num_classes; c++)
            {
                float sum = 0.0f;
                for (int k=0; k<num_features; k++)
                {
                    sum += weights_out[k*num_classes + c]*features[j*num_features+k];
                }
                logits[j*num_classes+c] = sum;
            }
        }
    }
    
    auto end_layer = chrono::high_resolution_clock::now();
    cout << "  -> Layer " << i+1 << " completato in " 
         << chrono::duration_cast<chrono::milliseconds>(end_layer - start_layer).count() 
         << " ms." << endl;
  }  
}

