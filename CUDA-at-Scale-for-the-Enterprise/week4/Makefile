IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -I/home/coder/lib/cub/ -I/home/coder/lib/cuda-samples/Common -lcudart -lcuda --std c++17

.PHONY: clean build run

build: *.cu *.h
	$(COMPILER) $(COMPILER_FLAGS) *.cu -o merge_sort.exe

clean:
	rm -f merge_sort.exe

run:
	./merge_sort.exe $(ARGS)

all: clean build run
