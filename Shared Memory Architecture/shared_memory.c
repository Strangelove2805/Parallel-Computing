#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

int matrix_size = 10;
int thread_count = 4;
double precision = 0.01;
double** relax_matrix;
pthread_mutex_t** mutex_matrix;


double** relax_memory(double** matrix, int matrix_size) {

	// The matrix size is variable, the memory for it must be allocated
	matrix = malloc(matrix_size * sizeof(double*));
	// Avoid dereferencing a null pointer, causes undefined behaviour
	if (matrix == NULL) return NULL;

	for (int i = 0; i < matrix_size; i++) {
		if (matrix == NULL) return NULL;
		matrix[i] = malloc(matrix_size * sizeof(double));
	}
	return matrix;
}


pthread_mutex_t** mutex_memory(pthread_mutex_t** matrix, int matrix_size) {

	// Allocate a matrix of mutex locks of dimension size-2 such that
	// the locks only cover the variable part of the relax matrix
	matrix = malloc(matrix_size * sizeof(pthread_mutex_t*));
	if (matrix == NULL) return NULL;

	for (int i = 0; i < matrix_size; i++) {
		if (matrix == NULL) return NULL;
		matrix[i] = malloc(matrix_size * sizeof(pthread_mutex_t));
	}
	return matrix;
}


double** build_relax() {

	// Assign zeroes to all matrix cells except the boundaries
	for (int i = 0; i < matrix_size; i++) {
		for (int j = 0; j < matrix_size; j++) {
			if (i == 0 || j == 0 || i == (matrix_size - 1) || j == (matrix_size - 1)) {
				if (relax_matrix[i] == NULL) return NULL;
				// Assign ones to all boundary cells
				relax_matrix[i][j] = (double)(1);
			}
			else {
				relax_matrix[i][j] = (double)(0);
			}
		}
	}
	return relax_matrix;
}


pthread_mutex_t** build_mutex() {

	// Initialise a mutex lock for each cell in the mutex matrix
	for (int i = 0; i < matrix_size - 2; i++) {
		for (int j = 0; j < matrix_size - 2; j++) {

			if (relax_matrix[i] == NULL) return NULL;
			pthread_mutex_init(&mutex_matrix[i][j], NULL);

		}
	}
	return mutex_matrix;
}


// Input arguments must be of type void* for the pthreads
// The relaxation process is algorithmically similar to a sequential
// implementation, moving from the top left of the matrix to the bottom
// right (excluding edge cells), except when a thread is modifying a value, 
// that cell is locked and other threads will skip over it
void* relaxation(void* args) {

	while (1 == 1) {

		// Counter to track the number of differences above precision
		int precision_counter = 0;

		// Loop from 1 to size-1, avoiding the matrix edges
		for (int i = 1; i < matrix_size - 1; i++) {

			for (int j = 1; j < matrix_size - 1; j++) {

				// Mutex trylock will attempt to lock a mutex. If that mutex
				// is already locked, the function will return -1, at which
				// point the thread will skip over the current matrix cell onto the 
				// next loop iteration. If the mutex is unlocked, this function will
				// lock it and the thread will move into the critical section.
				// We use i-1 and j-1 because the mutex matrix is 2-smaller in
				// both dimensions than the relax matrix, so to link a mutex
				// to its corresponding value we must shift right 1 and down 1
				if (pthread_mutex_trylock(&mutex_matrix[i - 1][j - 1]) != 0) {
					continue;
				}

				// Grab the current value
				double current = relax_matrix[i][j];

				// Calculate the average of the 4 adjacent cells
				relax_matrix[i][j] = (relax_matrix[i - 1][j] + relax_matrix[i + 1][j]
					+ relax_matrix[i][j - 1] + relax_matrix[i][j + 1]) / 4;

				// Unlock the mutex corresponding to the just-changed cell
				pthread_mutex_unlock(&mutex_matrix[i - 1][j - 1]);

				// Find the difference between old and new values
				double delta = (double)fabs(current - relax_matrix[i][j]);

				if (delta > precision) {
					precision_counter++;
				}
			}
		}
		// If all cell differences are within precision, problem solved
		if (precision_counter == 0) break;
	}
	// Once precision is reached, kill the thread
	pthread_exit(0);
}



int main() {

	struct timespec start, end;
	
	// Allocate the memory for and build the matrix to work on
	relax_matrix = relax_memory(relax_matrix, matrix_size);
	relax_matrix = build_relax();

	// Allocate the memory for and build the matrix of mutex locks
	mutex_matrix = mutex_memory(mutex_matrix, matrix_size - 2);
	mutex_matrix = build_mutex();

	clock_gettime(CLOCK_MONOTONIC, &start);
	// Create a structure to catalogue thread IDs and prevent races
	struct thread_ids {int thread_identity;};
	// Initialise the threads we need in an array
	pthread_t threads[thread_count];
	// Create the thread ID catalogue, size n for n threads
	struct thread_ids identifiers[thread_count];

	for (int i = 0; i < thread_count; i++) {
		// Assign each thread a unique identity based on i
		identifiers[i].thread_identity = i;
		// Create a NULL attribute thread i, with ID i, and run relaxation
		pthread_create(&threads[i], NULL, relaxation, &identifiers[i]);
	}

	// Wait for all threads to finish the relaxation process and join them
	for (int i = 0; i < thread_count; i++) {
		pthread_join(threads[i], NULL);
	}
	
	clock_gettime(CLOCK_MONOTONIC, &end);
	
	double time_taken;
    	time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  
   	printf("%lf", time_taken);

	// Free up the memory for the matrices used
	free(relax_matrix);
	free(mutex_matrix);

}