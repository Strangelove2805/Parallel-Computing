#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


// Perform relaxation on a given cell in the read-only buffer, place the result in
// the write-buffer. Determine whether the result is not within precision - 
// return True (1) if so, which will increment the precision flag for the current 
// child process
int relaxation(double* read_from, double* write_to, int i, int matrix_size, double precision) {

    // Sum adjacent cells, cells in the row above and below are i +/- one full row length
    double sum = (read_from[i - matrix_size] +
        read_from[i + matrix_size] +
        read_from[i - 1] +
        read_from[i + 1]) / 4; // Divide by 4

    // Store unrelaxed value presently at i, write new relaxed value to the write buffer
    double current = read_from[i];
    write_to[i - matrix_size] = sum;

    return (precision < fabs(current - sum));

}


int main() {

    // For calculating the program runtime
    double starttime, endtime;

    // Ensuring the matrix dimensions and precision cannot be altered between processes
    const int matrix_size = 1000;
    const double precision = 0.01;

    // Keep a record of the number of cells contained in the rows above and below a row
    // being worked on, for when we create the 3-row buffer we will read from
    const int read_only_rows = 2 * matrix_size;

    // 1D array representing an NxN 2D matrix
    double* relax_matrix = NULL;

    // INITIALISE MPI ENVIRONMENT:

    MPI_Init(NULL, NULL);

    // Identification for a given process
    int process_rank;

    // Total number of processes
    int process_total;

    // Get current process rank and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_total);

    // Start recording runtime
    starttime = MPI_Wtime();

    // FILL RELAX MATRIX:

    // Matrix should be filled with zeroes, borders should be ones. This allows
    // us to easily compare the results to a control case made using the
    // sequential version of the program, and since I used the same matrix values
    // for coursework 1, the runtimes can be compared since there will be no
    // uncontrollable fluctuations caused by having random numbers in the cells,
    // e.g a case where precision is reached in only a few iterations

    // Fill the matrix with its starting values, only during the parent process
    if (process_rank == 0) {

        // Allocate memory for the matrix
        relax_matrix = malloc(matrix_size * matrix_size * sizeof(double));
        if (relax_matrix == NULL) return 1;

        for (int i = 0; i < matrix_size * matrix_size; ++i) {

            // If cell i is a matrix border cell, give it a value of 1
            if (i < matrix_size
                || i + matrix_size >= (matrix_size * matrix_size)
                || i % matrix_size == 0
                || (i + 1) % matrix_size == 0) {

                relax_matrix[i] = 1;

            }

            // Fill non-border cells with 0s
            else {

                relax_matrix[i] = 0;

            }
        }
    }

    // ROW / PROCESS ALLOCATION:

    // Determine the number of cells the processes will handle
    const int process_allocation = matrix_size * ((matrix_size - 2) / process_total);

    // Consider awkward matrix sizes - define which processes 
    // will handle additional "odd" rows
    const int odd_row = (matrix_size - 2) % process_total;

    // Allocate memory for counts and column-representing offsets 
    // used when scattering matrix chunks
    int* send_counts = malloc(process_total * sizeof(int));
    int* send_offsets = malloc(process_total * sizeof(int));
    if (send_counts == NULL || send_offsets == NULL) return 1;

    // Do the same as above, but for counts and offsets used in
    // the gathering of chunks
    int* receive_counts = malloc(process_total * sizeof(int));
    int* receive_offsets = malloc(process_total * sizeof(int));
    if (receive_counts == NULL || receive_offsets == NULL) return 1;

    // For each of the processes...
    for (int process = 0; process < process_total; ++process) {

        // In the case of those awkward processes that must handle an odd
        // row, add an additional matrix_size (1 row) to the allocated cells
        if (process < odd_row) {

            send_counts[process] = process_allocation
                + read_only_rows
                + matrix_size;

        }

        // Allocate the rows a normal-case process will start with, as well as
        // the rows above and below
        else {

            send_counts[process] = process_allocation
                + read_only_rows;

        }

        // Consider the offset resulting from the column number.
        // Not an issue for the first row
        if (process == 0) {

            send_offsets[process] = receive_counts[process - 1];

        }

        else {

            send_offsets[process] = receive_counts[process - 1]
                + send_offsets[process - 1];

        }

        // Write buffer can be two rows less than the send buffer as these
        // rows are not relaxed
        receive_counts[process] = send_counts[process] - read_only_rows;
        receive_offsets[process] = send_offsets[process] + matrix_size;

    }

    // MESSAGE PASSING & RELAXATION:

    // Generate the buffer that will hold a scattered portion of the matrix, plus
    // two additional rows
    double* read_buffer = malloc(sizeof(double) * send_counts[process_rank]);
    if (read_buffer == NULL) return 1;

    // Generate the buffer that relaxed values will be written to, then gathered
    double* write_buffer = malloc(sizeof(double) * receive_counts[process_rank]);
    if (write_buffer == NULL) return 1;

    // Counters to check that precision is reached:
    // Child flag  - Precision has been reached in a given process if this
    //               value == 0, i.e all relaxed values are within precision
    // Parent flag - Reduction of the child flags across all processes tells
    //               us if another relax iteration is required. If all chunks
    //               across all processes are within precision, this will be 0
    int precision_flag_child = 0;
    int precision_flag_parent = 0;

    while (1 == 1) {

        // Scatter relax matrix into chunks using our counts for each process
        // with the 3-row long read_buffer as our receive buffer. Chunks may differ 
        // in size owing to the presence of the odd rows.
        MPI_Scatterv(relax_matrix, send_counts, send_offsets, MPI_DOUBLE,
            read_buffer, send_counts[process_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Run from the second row to the second-to-last row. The first and last rows
        // are constant so iterating over them is pointless.
        for (int i = matrix_size; i < (matrix_size + receive_counts[process_rank]); ++i) {

            // Using a 1D array in place of a 2D matrix means factoring the column
            // number into our single cell index, i. This is simply the sum of our
            // position within the current row and the column offset for this process
            int column_factored_index = receive_offsets[process_rank] + i - matrix_size;

            if (column_factored_index < matrix_size
                || column_factored_index + matrix_size >= (matrix_size * matrix_size)
                || column_factored_index % matrix_size == 0
                || (column_factored_index + 1) % matrix_size == 0) {

                write_buffer[i - matrix_size] = read_buffer[i];

            }
            // Otherwise, relax the cell and determine if it's within precision.
            else {

                precision_flag_child += relaxation(read_buffer,
                    write_buffer,
                    i,
                    matrix_size,
                    precision);

            }
        }

        // Gather the matrix chunks across all child processes to the relax array 
        // in the parent
        MPI_Gatherv(write_buffer, receive_counts[process_rank], MPI_DOUBLE, relax_matrix,
            receive_counts, receive_offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Reduce the precision flags across all child processes to the parent
        MPI_Allreduce(&precision_flag_child, &precision_flag_parent,
            1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);

        // If there are still child processes not in precision, 
        // reset the flags and keep relaxing
        if (precision_flag_parent != 0) {

            precision_flag_child = 0;

        }

        else {

            break;

        }
    }

    // DEALLOCATE MEMORY & FINALIZE MPI

    if (process_rank == 0) {

        free(relax_matrix);
        free(send_counts);
        free(receive_counts);
        free(send_offsets);
        free(receive_offsets);

    }

    free(read_buffer);
    free(write_buffer);

    MPI_Barrier(MPI_COMM_WORLD);

    endtime = MPI_Wtime();

    MPI_Finalize();

    // Print total runtime (not combined runtime of processes)
    if (process_rank == 0) {

        printf("Runtime = %f\n", endtime - starttime);

    }

    return 0;

}