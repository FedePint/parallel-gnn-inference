#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "bin_conversion.h"
#include "random_weights_matrix.h"
#include "globals.h"
#include <nvtx3/nvToolsExt.h>


using namespace std;


unordered_map<string, int> meta;
vector<float> features;
vector<int> indices;
vector<int> indptr;
vector<vector<float>> layer_weights;
vector<float> weights_out;
vector<float> next_features;
vector<float> logits;
vector<int> edge_dest;

string select_dataset() {
    int choice;
    cout << "Seleziona il dataset da utilizzare:\n";
    cout << "1. ogbn-arxiv (Reale)\n";
    cout << "2. scale-free (Sintetico)\n";
    cout << "3. random (Sintetico)\n";
    cout << "4. small-world (Sintetico)\n";
    cout << "5. synthetic_small-world_128 (Sintetico)\n";
    cout << "6. synthetic_small-world_256 (Sintetico)\n";
    cout << "7. synthetic_small-world_32 (Sintetico)\n";
    cout << "8. synthetic_small-world_64 (Sintetico)\n";
    cout << "9. synthetic_scale-free_250000 (Sintetico)\n";
    cout << "10. synthetic_scale-free_500000 (Sintetico)\n";
    cout << "11. synthetic_scale-free_1500000 (Sintetico)\n";
    cout << "12. synthetic_scale-free_2500000 (Sintetico)\n";
    cout << "Scelta: ";
    cin >> choice;

    switch(choice) {
        case 1: return "data/ogbn-arxiv_csr";
        case 2: return "data/synthetic_scale-free_csr";
        case 3: return "data/synthetic_random_csr";
        case 4: return "data/synthetic_small-world_csr";
        case 5: return "data/synthetic_small-world_128_csr";
        case 6: return "data/synthetic_small-world_256_csr";
        case 7: return "data/synthetic_small-world_32_csr";
        case 8: return "data/synthetic_small-world_64_csr";
        case 9: return "data/synthetic_scale-free_250000_csr";
        case 10: return "data/synthetic_scale-free_500000_csr";
        case 11: return "data/synthetic_scale-free_1500000_csr";
        case 12: return "data/synthetic_scale-free_2500000_csr";
        default: 
            cout << "Scelta non valida, uso ogbn-arxiv di default.\n";
            return "data/ogbn-arxiv_csr";
    }
}

vector<int> edge_dest_compute(const vector<int>& indices, const vector<int>& indptr)
{
    vector<int> edge_dest(indices.size());
    for (int i=0; i<meta["num_nodes"]; i++)
    {
        for (int j=indptr[i]; j<indptr[i+1]; j++)
        {
            edge_dest[j] = i;
        }
    }
    return edge_dest;
}

// menù di selezione algoritmo
int select_algorithm() {
    int choice;
    cout << "Seleziona l'algoritmo da eseguire:\n";
    cout << "1. Sequenziale\n";
    cout << "2. Parallelismo sui Nodi\n";
    cout << "3. Parallelismo sui chunks di Nodi\n";
    cout << "4. Parallelismo sugli Archi\n";
    cout << "5. CUDA (Thread per Nodo)\n";
    cout << "6. CUDA (Thread per Feature)\n";
    cout << "7. CUDA (Thread per Feature con Compressione FP16)\n";
    cout << "8. CUDA (Thread per Edge)\n";
    cout << "Scelta: ";
    cin >> choice;
    return choice;
}

float compute_accuracy(const vector<float>& logits, const vector<int>& labels, const vector<int>& split_indices, int num_classes)
{
    int correct=0;
    int predicted = -1;
    for (int ind:split_indices)
    {
        float max = -1e30f;
        for(int i=0; i<num_classes; i++)
        {
            if (logits[ind*num_classes+i]>max)
            {
                max=logits[ind*num_classes+i];
                predicted = i;
                
            }
        }
        if (predicted==labels[ind])
        {
            correct++;
        }
    }
    return ((float)correct/split_indices.size())* 100.0f;
}

int main(){
    string dataset_path = select_dataset();
    int algorithm_choice = select_algorithm();
    
    cout << "[Checkpoint] Caricamento dataset da: " << dataset_path << " ..." << endl;
    meta = read_from_meta(dataset_path);
    features = read_from_features(dataset_path);
    indices = read_from_indices(dataset_path);
    indptr = read_from_indptr(dataset_path);
    layer_weights.resize(NUM_LAYERS - 1);
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        layer_weights[i] = generate_random_weights(meta["num_features"], meta["num_features"]);
    }
    weights_out = generate_random_weights(meta["num_features"], meta["num_classes"]);
    logits.resize(meta["num_nodes"] * meta["num_classes"]);

    // calcolo qui vettore edge così non allungo tempo di inferenza
    cout << "[Checkpoint] Calcolo array edge_dest..." << endl;
    edge_dest = edge_dest_compute(indices, indptr);

    
    

    auto start_total = chrono::high_resolution_clock::now();
    
    cout << "[Checkpoint] Inizio calcolo inferenza GNN..." << endl;
    auto start_inf = chrono::high_resolution_clock::now();
    
    // esecuzione dell'algoritmo scelto
    if (algorithm_choice == 1) {
        cout << "[Info] Esecuzione in modalita' Sequenziale" << endl;

        nvtxRangePushA("Inference Engine Sequential");
        inference_engine_sequential(features, indices, indptr);
        nvtxRangePop();
    } else if (algorithm_choice == 2) {
        cout << "[Info] Esecuzione in modalita' Parallelismo sui Nodi" << endl;

        nvtxRangePushA("Inference Engine node Parallel");
        inference_engine_node();
        nvtxRangePop();
    } else if (algorithm_choice == 3) {
        cout << "[Info] Esecuzione in modalita' Parallelismo sui chunks di Nodi" << endl;

        nvtxRangePushA("Inference Engine node Parallel chunk");
       
        inference_engine_node_chunk();

        nvtxRangePop();
    } else if (algorithm_choice == 4) {
        cout << "[Info] Esecuzione in modalita' Parallelismo sugli Archi" << endl;
        nvtxRangePushA("Inference Engine Edge Parallel");
        
        inference_engine_edge_node_chunk();
        
        // NVTX per avere tempo di inferenza
        nvtxRangePop();
        
    } else if (algorithm_choice == 5) {
        cout << "[Info] Esecuzione in modalita' CUDA (Thread per Nodo)" << endl;
        inference_engine_cuda_node();
    } else if (algorithm_choice == 6) {
        cout << "[Info] Esecuzione in modalita' CUDA (Thread per Feature)" << endl;
        inference_engine_cuda_thread_per_feature();
    } else if (algorithm_choice == 7) {
        cout << "[Info] Esecuzione in modalita' CUDA (Thread per Feature con Compressione FP16)" << endl;
        inference_engine_cuda_thread_per_feature_compressed();
    } else if (algorithm_choice == 8) {
        cout << "[Info] Esecuzione in modalita' CUDA (Thread per Edge)" << endl;
        inference_engine_cuda_thread_per_edge();
    } else {
        cout << "[Errore] Scelta non valida o algoritmo non implementato!" << endl;
        return 1;
    }
    
    auto end_inf = chrono::high_resolution_clock::now();
    cout << "[Timer] Inferenza totale completata in " 
         << chrono::duration_cast<chrono::milliseconds>(end_inf - start_inf).count() 
         << " ms.\n" << endl;

    cout << "[Checkpoint] Caricamento labels e indici del test set..." << endl;
    auto start_load = chrono::high_resolution_clock::now();
    
    vector<int> labels = read_from_labels(dataset_path);
    vector<int> test_idx = read_from_test_split(dataset_path);
    auto end_load = chrono::high_resolution_clock::now();
    cout << "[Timer] Caricamento file binari completato in " 
         << chrono::duration_cast<chrono::milliseconds>(end_load - start_load).count() 
         << " ms.\n" << endl;


         
    cout << "[Checkpoint] Calcolo accuratezza sul test set..." << endl;
    auto start_acc = chrono::high_resolution_clock::now();
    
    float test_accuracy = compute_accuracy(logits, labels, test_idx, meta["num_classes"]);
    auto end_acc = chrono::high_resolution_clock::now();
    cout << "[Timer] Accuratezza calcolata in " 
         << chrono::duration_cast<chrono::milliseconds>(end_acc - start_acc).count() 
         << " ms.\n" << endl;

    auto end_total = chrono::high_resolution_clock::now();

    cout << "--- Valutazione Test Set ---" << endl;
    cout << "Numero nodi di test:  " << test_idx.size() << endl;
    cout << "Test Accuracy:        " << test_accuracy << "%" << endl;
    cout << "Tempo totale (main):  " 
         << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() 
         << " ms." << endl;

    return 0;
}
