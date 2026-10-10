    #ifndef BIN_CONVERSION_H
#define BIN_CONVERSION_H

#include <string>
#include <unordered_map>
#include <vector>

using namespace std;


unordered_map<string, int> read_from_meta(const string& base_path);


vector<float> read_from_features(const string& base_path);


vector<int> read_from_indices(const string& base_path);


vector<int> read_from_indptr(const string& base_path);


vector<int> read_from_labels(const string& base_path);


vector<int> read_from_test_split(const string& base_path);

#endif // BIN_CONVERSION_H
