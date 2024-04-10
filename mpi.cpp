#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <mpi.h>

using namespace std::chrono;

// Defining the dimension of the square matrices
const int MATRIX_DIM = 1000;

const int NUM_THREADS = 1;

// Function to fill a matrix with random values
void fillMatrixWithRandomValues(int matrix[][MATRIX_DIM])
{
    for (int row = 0; row < MATRIX_DIM; ++row)
        for (int col = 0; col < MATRIX_DIM; ++col)
            matrix[row][col] = rand() % 100; // Random value between 0 and 99
}

// Function to calculate the product of two matrices
void calculateProductOfMatrices(const int firstMatrix[][MATRIX_DIM], const int secondMatrix[][MATRIX_DIM], int productMatrix[][MATRIX_DIM], int startRow, int endRow)
{
    for (int row = startRow; row < endRow; ++row)
    {
        for (int col = 0; col < MATRIX_DIM; ++col)
        {
            productMatrix[row][col] = 0;
            for (int k = 0; k < MATRIX_DIM; ++k)
            {
                productMatrix[row][col] += firstMatrix[row][k] * secondMatrix[k][col];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int size, rank; // Number of processes and rank of the current process

    // Initializing the MPI environment
    MPI_Init(&argc, &argv);

    // Getting the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Getting the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Seeding the random number generator differently for each process
    srand(time(nullptr) + rank);

    // Declaring matrices
    int firstMatrix[MATRIX_DIM][MATRIX_DIM];
    int secondMatrix[MATRIX_DIM][MATRIX_DIM];
    int productMatrix[MATRIX_DIM][MATRIX_DIM] = {0};

    // Master process (rank 0) fills the input matrices
    if (rank == 0)
    {
        fillMatrixWithRandomValues(firstMatrix);
        fillMatrixWithRandomValues(secondMatrix);
    }

    // Broadcasting the second matrix to all processes
    MPI_Bcast(&secondMatrix, MATRIX_DIM * MATRIX_DIM, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculating the number of rows each process should handle
    int rowsPerProcess = MATRIX_DIM / size;
    int startRow = rank * rowsPerProcess;     // Starting row for the current process
    int endRow = (rank + 1) * rowsPerProcess; // Ending row for the current process

    // Scattering the first matrix rows among processes
    MPI_Scatter(firstMatrix, rowsPerProcess * MATRIX_DIM, MPI_INT, &firstMatrix[startRow], rowsPerProcess * MATRIX_DIM, MPI_INT, 0, MPI_COMM_WORLD);

    // Starting measuring the computation time
    auto startMoment = high_resolution_clock::now();

    // Each process computes its portion of the matrix product
    calculateProductOfMatrices(firstMatrix, secondMatrix, productMatrix, startRow, endRow);

    // End measuring the computation time
    auto endMoment = high_resolution_clock::now();

    // Gathering the partial results from all processes to the master process
    MPI_Gather(&productMatrix[startRow], rowsPerProcess * MATRIX_DIM, MPI_INT, productMatrix, rowsPerProcess * MATRIX_DIM, MPI_INT, 0, MPI_COMM_WORLD);

    // Master process (rank 0) prints the output
    if (rank == 0)
    {
        auto timeTaken = duration_cast<seconds>(endMoment - startMoment);

        cout << "MPI Matrix Multiplication Performance" << endl;
        cout << "Dimension: " << MATRIX_DIM << endl;
        cout << "Process: " << size << endl;
        cout << "Threads: " << NUM_THREADS << endl;
        cout << "Runtime: " << timeTaken << endl;
    }

    // Finalizing the MPI environment
    MPI_Finalize();
    return 0;
}
