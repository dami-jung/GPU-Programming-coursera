week4

For ```helper_cuda.h: No such file or directory``` error,
```
cd ..
mkdir lib && cd lib
git clone https://github.com/NVIDIA/cuda-samples.git
cd ..
cd project
```

# Assignment Instructions

A good multiple language guide to the MergeSort algorithm can be found at the following Geeks for Geeks article: https://www.geeksforgeeks.org/merge-sort/

F​or this assignment, you will develop code that will implement the MergeSort algorithm. The code that is in the project folder, most specifically merge_sort.cu, holds the scaffolding for completing this task. There is copious amounts of comments to make development easier and you can search for TODO  in the code for more specific direction of what to implement where. Below are explanations for the various buttons at the bottom of the browser, use them to develop, build, and execute your code accordingly:

Clean old builds and output files - This button will remove all executable and output files from the project folder

Build Code - Builds the merge_sort.cu file into the merge_sort.exe executable file.

R​un Tests - Executes a prescribed run of the merge sort algorithm

Submit Assignment - Submits the results of executing the tests (output.txt) for grading

N​ote that you should not remove any print statements or output to files.

T​he command line arguments for the executable merge_sort.exe will be:

- -​x Threads Per Block in x dimension

- -y Threads Per Block in y dimension

- -z Threads Per Block in z dimension

- -X Blocks per Grid in x dimension

- -Y Blocks per Grid in y dimension

- -Z Blocks per Grid in z dimension

- -​n number of elements in the randomly generated input data, defaults to 32

Y​our code should be able to handle a large n, 1000s, and various configurations for the other argume
