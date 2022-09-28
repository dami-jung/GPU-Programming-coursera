# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: cublas_assignment.cu
	$(CXX) cublas_assignment.cu --std c++17 `pkg-config opencv --cflags --libs` -o cublas_assignment.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcublas

run:
	./cublas_assignment.exe > output.txt

clean:
	rm -f cublas_assignment.exe output.txt 