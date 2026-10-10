CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++20 -I./OGB_extraction -Wall -g
NVCCFLAGS = -O3 -std=c++17 -I./OGB_extraction -w -lineinfo --compiler-options -D_FORTIFY_SOURCE=0
LDFLAGS = -lpthread -L/usr/local/cuda/lib64 -lcudart -lnvToolsExt

TARGET = gnn_inference

# Tutti i file sorgente C++
CPP_SRCS = main.cpp \
           sequential.cpp \
           node_parallelism.cpp \
           node_parallelism_chunk.cpp \
           edge_parallelism_chunk.cpp \
           random_weights_matrix.cpp \
           OGB_extraction/bin_conversion.cpp

# Tutti i file sorgente CUDA
CU_SRCS = thread_per_node.cu \
          thread_per_feature.cu \
          thread_per_feature_compressed.cu \
          thread_per_edge.cu

# Lista automatica degli oggetti
OBJS = $(CPP_SRCS:.cpp=.o) $(CU_SRCS:.cu=.o)

# Regola principale
all: $(TARGET)

# Link dell'eseguibile usando g++ e linkando le librerie CUDA
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compilazione file .cpp con g++ in C++20 (necessario per std::atomic_ref)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compilazione file .cu con nvcc in C++17 (per evitare bug di CUDA con gli header C++20)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Pulizia dei file compilati
clean:
	rm -f *.o OGB_extraction/*.o $(TARGET)
