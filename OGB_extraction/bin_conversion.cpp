#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// legge meta.txt e restituisce un dizionario
unordered_map<string, int> read_from_meta(const string& base_path) {
    unordered_map<string, int> meta_dict;
    ifstream file(base_path + "/meta.txt");
    
    if (!file.is_open()) {
        cerr << "Errore nell'apertura del file: meta.txt" << endl;
        return meta_dict;
    }

    string key;
    int value;
    // legge automaticamente la parola (chiave) e il numero (valore) per ogni riga
    while (file >> key >> value) {
        meta_dict[key] = value;
    }

    file.close();
    return meta_dict;
}




//legge da features.bin e restituisce un vettore 1D di float
vector<float> read_from_features(const string& base_path) {
    vector<float> features;

    ifstream file(base_path + "/features.bin", ios::binary);

    if (!file.is_open()) {
        cerr << "Errore nell'apertura del file: features.bin" << endl;
        return features;
    }
    unordered_map<string, int> meta = read_from_meta(base_path);

    for (int i = 0; i < meta["num_nodes"]*meta["num_features"]; i++) {
        float feature;
        file.read((char*)&feature, sizeof(feature));
        features.push_back(feature);
    }

    file.close();
    return features;

}
    
// legge da indici.bin
vector<int> read_from_indices(const string& base_path){
    vector<int> indices;
    ifstream file(base_path + "/indices.bin", ios::binary);

    if (!file.is_open()) {
        cerr << "Errore nell'apertura del file: indices.bin" << endl;
        return indices;
    }

    int index;
    while(file.read((char*)&index, sizeof(int))){
        indices.push_back(index);
    }

    file.close();
    return indices;
}


//legge indptr.bin 
vector<int> read_from_indptr(const string& base_path){
    vector<int> indptr;
    ifstream file(base_path + "/indptr.bin", ios::binary);
    
    if (!file.is_open()) {
        cerr << "Errore nell'apertura del file: indptr.bin" << endl;
        return indptr;
    }

    int index;
    while(file.read((char*)&index, sizeof(int))){
        indptr.push_back(index);
    }

    file.close();
    return indptr;
}

vector<int> read_from_labels(const string& base_path){
    vector<int> labels;
    ifstream file(base_path + "/labels.bin", ios::binary);
    
    if (!file.is_open()) {
        cerr << "Errore nell'apertura del file: labels.bin" << endl;
        return labels;
    }

    int label;
    while(file.read((char*)&label, sizeof(int))){
        labels.push_back(label);
    }

    file.close();
    return labels;
}

vector<int> read_from_test_split(const string& base_path){
    vector<int> test_split;
    ifstream file(base_path + "/test_idx.bin", ios::binary);
    
    if (!file.is_open()) {
        cerr << "Errore nell'apertura del file: test_idx.bin" << endl;
        return test_split;
    }

    int test_idx;
    while(file.read((char*)&test_idx, sizeof(int))){
        test_split.push_back(test_idx);
    }

    file.close();
    return test_split;
}

