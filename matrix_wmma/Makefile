CUDA_HOME=$(shell echo ~)/local/cuda-10.0/
NVCC=$(CUDA_HOME)/bin/nvcc
NVPROF=$(CUDA_HOME)/bin/nvprof
NVCCFLAGS=-std=c++14 -arch=sm_70 -Xptxas -dlcm=cg
TARGET=matrix_wmma

all:$(TARGET)


$(TARGET):main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

test:
	./$(TARGET)

prof:
	$(NVPROF) ./$(TARGET) > /dev/null 2> log.txt
	$(NVPROF) --print-gpu-trace ./$(TARGET)  > /dev/null 2>> log.txt 
	cat log.txt
