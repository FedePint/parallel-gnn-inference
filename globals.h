#ifndef GLOBALS_H
#define GLOBALS_H

#include <vector>
#include <unordered_map>
#include <string>

extern std::unordered_map<std::string, int> meta;
extern std::vector<float> features;
extern std::vector<int> indices;
extern std::vector<int> indptr;
extern std::vector<std::vector<float>> layer_weights;
extern std::vector<float> weights_out;
extern std::vector<float> next_features;
extern std::vector<float> logits;
extern std::vector<int> edge_dest;

const int NUM_LAYERS = 3;


std::vector<int> edge_dest_compute(const std::vector<int>& indices, const std::vector<int>& indptr);
void inference_engine_sequential(const std::vector<float>& features_in, const std::vector<int>& indices, const std::vector<int>& indptr);
void inference_engine_node();
void inference_engine_node_chunk();
void inference_engine_edge_node_chunk();
void inference_engine_cuda_node();
void inference_engine_cuda_thread_per_feature();
void inference_engine_cuda_thread_per_feature_compressed();
void inference_engine_cuda_thread_per_edge();


#endif
