#include <iostream>
#include <string>
#include <queue>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "bin_conversion.h"
#include "random_weights_matrix.h"

using namespace std;

#include "globals.h"

class ThreadPool{
 
    private:
        vector<thread> workers;
        queue<function<void()>> tasks;
        bool stop=false;
        mutex mtx;
        condition_variable cv;
    
    public:
        ThreadPool()
        {
            for (int i=0; i<thread::hardware_concurrency(); i++)
            {
                workers.emplace_back([this] (){
                    while (true)
                    {
                        function<void()> task;
                        {
                            unique_lock<mutex> Lock(this->mtx);
                            this->cv.wait(Lock, [this](){ return this->stop || !this->tasks.empty();});
                            if (this->stop && this->tasks.empty())
                            {
                                return;
                            }

                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                            
                        }
                        task();
                    }
                });
            }
        }

        void enqueue(function<void()> task)
        {
            {
               unique_lock<mutex> Lock(this->mtx);
               this->tasks.push(task);
            }
            this->cv.notify_one();
        }

        ~ThreadPool()
        {
            {
                unique_lock<mutex> Lock(this->mtx);
                this->stop = true;
            }
            this->cv.notify_all();
            for (thread &worker : this->workers)
            {
                worker.join();
            }
        }


};


static mutex layer_mtx;
static condition_variable layer_cv;
static atomic<int> nodes_computed(0);

void workerNode(int j, const vector<float>& weights)
{
            int num_features = meta["num_features"];

            vector<float> vsum(num_features, 0.0f);
            
            // aggiungo nodo stesso
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

            int completati = ++nodes_computed;
            if (completati == meta["num_nodes"]) {
                lock_guard<mutex> Lock(layer_mtx);
                layer_cv.notify_one();
            }
        
}

void workerFinalLayerNode(int j)
{
            int num_features = meta["num_features"];
            int num_classes = meta["num_classes"];
            
     for (int c=0; c<num_classes; c++)
            {
                float sum = 0.0f;
                for (int k=0; k<num_features; k++)
                {
                    sum += weights_out[k*num_classes + c]*features[j*num_features+k];
                }
                logits[j*num_classes+c] = sum;
            }
    
    int completati = ++nodes_computed;
    if (completati == meta["num_nodes"]) {
        lock_guard<mutex> Lock(layer_mtx);
        layer_cv.notify_one();
    }
}

void inference_engine_node()
{
  ThreadPool pool;

  for (int i=0; i<NUM_LAYERS; i++)
  {
    cout << "[Checkpoint] Avvio Layer " << i+1 << "..." << endl;
    auto start_layer = chrono::high_resolution_clock::now();
    
    next_features.clear();
    const vector<float>& weights = layer_weights[i];
    if (i<NUM_LAYERS-1) 
    {
        next_features.resize(meta["num_nodes"] * meta["num_features"]);
        
        for (int j=0; j<meta["num_nodes"]; j++)
        {
            pool.enqueue([&,j](){workerNode(j, weights);});
        }
    }
    else 
    {
        for (int j=0; j<meta["num_nodes"]; j++)
        {
            pool.enqueue([&,j](){workerFinalLayerNode(j);});
        }
    }

    {
        unique_lock<mutex> Lock(layer_mtx);
        layer_cv.wait(Lock, [&]() { return nodes_computed == meta["num_nodes"];});
        nodes_computed=0;
    }
    
    // Assegna next_features a features SOLO DOPO che tutti i thread hanno finito
    if (i<NUM_LAYERS-1) 
    {
        features = next_features;
    }
    
    auto end_layer = chrono::high_resolution_clock::now();
    cout << "  -> Layer " << i+1 << " completato in " 
         << chrono::duration_cast<chrono::milliseconds>(end_layer - start_layer).count() 
         << " ms." << endl;

  
    
  }  
}

