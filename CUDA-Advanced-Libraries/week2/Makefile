# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: cufft_example.cu
	$(CXX) cufft_example.cu --std c++17 `pkg-config opencv --cflags --libs` -o cufft_example.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I/usr/local/cuda/include -lcuda -lcufft

run:
	./cufft_example.exe $(ARGS) > output.txt

clean:
	rm -f cufft_example.exe output*.txt 