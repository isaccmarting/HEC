mpirun:
	mpirun -f machinefile ./main.exe
jmo:
	mpicc -fopenmp -o main.exe JacobiMPIandOpenMP.c -limf -lm
jm:
	mpicc -o main.exe JacobiMPI.c -limf -lm
js:
	mpicc -o main.exe JacobiSequential.c -limf -lm
clean:
	rm -f *.exe
