Sample: boxFilterNPP
Minimum spec: SM 2.0

A NPP CUDA Sample that demonstrates how to use NPP FilterBox function to perform a Box Filter.

Key concepts:
Performance Strategies
Image Processing
NPP Library

Y​ou will need to perform the following steps to develop new software that utilizes the CUDA cuFFT library:

F​ill in the stubbed out functions in the cufft_example.cu file, paying attention to the TODO and other comments placed in the functions to guide you through the development. It may also be helpful to look at your activity for this module to find similarities.

C​lick the Build button to build the code. If you encounter any issues skip to step 4.

I​f your code compiles, click the Run button to execute an iteration of your code with 256 elements, 16 x 16 matrix. If you want to run with more or less input values you can execute your code with make run ARGS="-n=16".

I​f you notice any issues, click the Clean button, which will clean any compiled files and output artifacts. Continue back to step 1.

I​f all succeeds and you would like to submit your assignment click the Submit Assignment button, which will submit your assignment for automatic grading.

N​otes: Feel free to include extra print lines for debugging or general clarity, but do not remove any predefined print lines. Also, if you want to explore different data types, take the opportunity to expand the assignment to allow input files (either by adding non-required arguments or hard coding input files). This should not break the Run command though. If you are so interested in FFTs and cuFFT in particular, also consider using this as a place to start exploring your capstone project, but it might make sense to submit with everything working, get the grade that you want and then modify the code.