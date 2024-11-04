#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

int matrix_size = 1000;
double precision = 0.01;
double** relax_matrix;


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


void relaxation(double precision, int matrix_size) {

	while (1 == 1) {

		// Counter to track the number of differences above precision
		int precision_counter = 0;

		// Loop from 1 to size-1, avoiding the matrix edges
		for (int i = 1; i < matrix_size - 1; i++) {

			for (int j = 1; j < matrix_size - 1; j++) {

				// Grab the current value
				double current = relax_matrix[i][j];

				// Calculate the average of the 4 adjacent cells
				relax_matrix[i][j] = (relax_matrix[i - 1][j] + relax_matrix[i + 1][j]
					+ relax_matrix[i][j - 1] + relax_matrix[i][j + 1]) / 4;

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
}


int main() {

	struct timespec start, end;
	// Allocate the memory for and build the matrix to work on
	relax_matrix = relax_memory(relax_matrix, matrix_size);
	relax_matrix = build_relax();

	clock_gettime(CLOCK_MONOTONIC, &start);
	// Modify the work matrix using the relaxation technique
	relaxation(precision, matrix_size);

	clock_gettime(CLOCK_MONOTONIC, &end);

	double time_taken;
	time_taken = (end.tv_sec - start.tv_sec) * 1e9;
	time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;

	printf("%lf", time_taken);

	// Free the memory for the work matrix
	free(relax_matrix);

}