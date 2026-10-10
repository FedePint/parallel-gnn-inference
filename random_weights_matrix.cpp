#include "random_weights_matrix.h"
#include <random>

using namespace std;

// Genera una matrice (rows x cols) di pesi casuali float in un array 1D
vector<float> generate_random_weights(int rows, int cols) {
    // 1. Alloca un vettore 1D di dimensione rows * cols
    vector<float> weights(rows * cols);
    
    
    // usando 'static', il generatore non si resetta ad ogni chiamata della funzione,
    // garantendo matrici diverse per layer diversi, ma sempre uguali ad ogni esecuzione del programma.
    static mt19937 gen(42);
    
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // popola il vettore 1D
    for (int i = 0; i < rows * cols; ++i) {
        weights[i] = dis(gen);
    }

    return weights;
}
