Compile and Run:
1. To compile the file, C++11 should be supported. The versin of gcc compiler should be at least 4.8. 

2. Before running any application, the user should check if the OpenMP version is newer than 3.1.
Setting CPU affinity can easily gain a 10X speedup.
This can be achieved simply by setting an environment variable in the
bashrc/bash_profile file:
  export OMP_PROC_BIND=true

3. Since both Scheduler and User Application Classes are templcate classes,
if any header file is modified, "make clean" is required before recompilation.

Description:
If the output layout is fixed, and the output array can be allocated in advance,
then feed the array address to the run/run2 function.
Otherwise, the user needs to transform a global combination map to a final output manually.
Global combination map is combination_map_ on the master node, and it can be retrieved by calling the function get_combination_map.

The following applications are provided:
1) Histogram;
2) K-Means;
3) Logistic Regression;
4) Window-Based Applications.

Histogram:
Make sure the value of STEP in simulation code is 1.

K-Means:
Make sure the value of STEP in simluation code and the value of NUM_DIMS in kmeans.h are equal.

Logistic Regression:
Make sure the value of STEP in simulation code and the value of NUM_COLS (NUM_DIMS + 1) in logistic_regression.h are equal.

Window-Based Applications:
To improve the efficiency by supporting early emission of reduction object in the RedObj class,
the trigger function should be overwritten. Moreover, the user should call the run2, not run function, to launch the data processing.

Possible improvements:
1. The key data type of reduction/combination map can be a string for better applicability,
but this increases the complexity of global synchrnoziation, since string-type data does not have a fixed size.
Keys of integer type can meet the requirements of most scientific analytics. 
2. Reduction object can be defined as protocol buffer,
to facilitate serializing variable-length data members (e.g., string type) in distributed environment.
