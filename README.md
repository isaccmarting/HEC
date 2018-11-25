# Codes which solve PDE with Jacobi method

## File Description
JacobiSequential.c: a serial implementation of Jacobi method.  
JacobiMPI.c: a parallel implementation of Jacobi method with pure MPI.  
JacobiMPIandOpenMP.c: a parallel implementation of Jacobi method with MPI+OpenMP.  
machinefile: the configuration file in the format "node: the number of processors".  
Makefile: the file containing required instructions to compile and run.  

## Requirements
1. MPI(My edition is 3.2.1); 
2. C Compiler spports OpenMP, such as gcc. 

## How to compile?
The instuctions are all in the Makefile.  
Compile JacobiSequential.c:   
　　make js  
and run it with `mpirun -np 1 ./main.exe`.  

compile JacobiMPI.c:  
　　make jm  
and run it wth `make mpirun`.  

compile JacobiMPIandOpenMP.c:  
　　make jmo  
and run it with `make mpirun`.  