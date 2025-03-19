CXX       = g++
NVCC      = nvcc
CXXFLAGS  = -O2 -fopenmp -Wall -std=c++17
NVCCFLAGS = -O2 -std=c++17 -Xcompiler "-Wall -fopenmp"
TARGET    = main
SRC       = main.cpp sorting.cpp
CU_SRC    = sorting_cuda.cu
HEADERS   = sorting.h particle.h

all: standard_sort

# Default build (uses std::sort)
standard_sort: $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Compile with Counting Sort
counting_sort: $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DUSE_COUNTING_SORT -o $(TARGET) $(SRC)

# Compile with Bitonic Sort
bitonic_sort: $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DUSE_BITONIC_SORT -o $(TARGET) $(SRC)

# Compile with Merge Sort
merge_sort: $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DUSE_MERGE_SORT -o $(TARGET) $(SRC)

# Compile with Bucket Sort
bucket_sort: $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DUSE_BUCKET_SORT -o $(TARGET) $(SRC)

counting_sort_cuda:$(SRC) $(CU_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -DUSE_COUNTING_SORT_CUDA -o $(TARGET) $(SRC) $(CU_SRC)

bitonic_sort_cuda:$(SRC) $(CU_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -DUSE_BITONIC_SORT_CUDA -o $(TARGET) $(SRC) $(CU_SRC)
clean:
	rm -f $(TARGET)




