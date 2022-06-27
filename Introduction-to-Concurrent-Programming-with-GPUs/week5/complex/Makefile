IDIR=./includes
EXECUTABLES_DIR=./bin
SOURCE_CODE_DIR=./includes
COMPILER=nvcc -rdc=true
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: $(SOURCE_CODE_DIR)/complex.cu
	mkdir $(EXECUTABLES_DIR)
	$(COMPILER) $(COMPILER_FLAGS) $(SOURCE_CODE_DIR)/complex.cu -o $(EXECUTABLES_DIR)/complex.exe

clean:
	rm -f complex.exe

run:
	$(EXECUTABLES_DIR)/complex.exe

all: clean build run
