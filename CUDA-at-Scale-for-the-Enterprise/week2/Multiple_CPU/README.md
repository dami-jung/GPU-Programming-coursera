# Assignment Instructions
T​his assignment requires you to develop code that compares a collection of floating point values sent to the CUDA code from two other CPU-based programs, written in Python 3. 

T​he steps that you will take to work on and then submit your code perform the following steps:

D​evelop your kernels that compares input vectors a and b, by index (compare a[i] to b[i], if a>b return 1, if a=b return 0, a<b return -1). Try to do this without explicitly using if-else branching, but points won't be taken off.

C​lick the Build button at the bottom of the web application, which will build the CUDA application that reads/writes data to/from the filesystem.

C​lick the Run Input A button, which executes a python script in a separate thread (CPU) that communicates with CUDA/GPU via the filesystem.

C​lick the Run Input B button, which executes a python script in a separate thread (CPU) that communicates with CUDA/GPU via the filesystem.

C​lick the Run Multiple CPU CUDA button, which start the CUDA program that determined the difference between the arrays of data that are sent via the file system and outputs their differences back to the filesystem. This may take a while and output a lot of files, do not touch anything until your CUDA program has completed.

C​lick the Create Submission File button, which creates the file that will be submitted for grading.

C​lick the Submit Assignment button, which will submit the above created file. If the submission works you will then want to go to the submission status page to see your submission status/grade.

N​ote: Please be very careful in modifying any code that outputs/prints to the filesystem or stdio. You can add debug statements but it would be best to remove them before submission. You should have 2 input CSV files (a and b) and 2 output csv files created as part of running the two python threads and the CUDA code.  The Create Submission File button, combines the 4 CSV files into a single CSV file with 4 rows.  Ensure that the results.csv file is a proper CSV that only has numeric values

*remove ₩n in output file to pass grader error
