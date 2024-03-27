#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <mpi.h> // Include MPI library

using namespace std::chrono;

const int MATRIX_DIM = 200;

void fillMatrixWithRandomValues(int matrix[][MATRIX_DIM]) {
    for (int row = 0; row < MATRIX_DIM; ++row)
        for (int col = 0; col < MATRIX_DIM; ++col)
            matrix[row][col] = rand() % 100;
}

// This function now takes the start and end row for computation
void calculateProductOfMatrices(const int firstMatrix[][MATRIX_DIM], const int secondMatrix[][MATRIX_DIM], int productMatrix[][MATRIX_DIM], int startRow, int endRow) {
    for (int row = startRow; row < endRow; ++row) {
        for (int col = 0; col < MATRIX_DIM; ++col) {
            productMatrix[row][col] = 0;
            for (int k = 0; k < MATRIX_DIM; ++k) {
                productMatrix[row][col] += firstMatrix[row][k] * secondMatrix[k][col];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int size, rank;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(nullptr) + rank); // Seed the random number generator differently for each process

    int firstMatrix[MATRIX_DIM][MATRIX_DIM];
    int secondMatrix[MATRIX_DIM][MATRIX_DIM];
    int productMatrix[MATRIX_DIM][MATRIX_DIM] = {0};

    // Master process
    if (rank == 0) {
        fillMatrixWithRandomValues(firstMatrix);
        fillMatrixWithRandomValues(secondMatrix);
    }

    // Broadcast second matrix to all processes
    MPI_Bcast(&secondMatrix, MATRIX_DIM*MATRIX_DIM, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the first matrix rows to different processes
    int rowsPerProcess = MATRIX_DIM / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank + 1) * rowsPerProcess;

    MPI_Scatter(firstMatrix, rowsPerProcess*MATRIX_DIM, MPI_INT, &firstMatrix[startRow], rowsPerProcess*MATRIX_DIM, MPI_INT, 0, MPI_COMM_WORLD);

    // Local computation
    auto startMoment = high_resolution_clock::now();
    calculateProductOfMatrices(firstMatrix, secondMatrix, productMatrix, startRow, endRow);
    auto endMoment = high_resolution_clock::now();

    // Gather results back to the master process
    MPI_Gather(&productMatrix[startRow], rowsPerProcess*MATRIX_DIM, MPI_INT, productMatrix, rowsPerProcess*MATRIX_DIM, MPI_INT, 0, MPI_COMM_WORLD);

    // Master process outputs the result
    if (rank == 0) {
        auto timeTaken = duration_cast<microseconds>(endMoment - startMoment);
        std::cout << "Multiplication completed in: " << timeTaken.count() << " microseconds" << std::endl;
        std::ofstream resultFile("Result_matrix.txt");
        for (int row = 0; row < MATRIX_DIM; ++row) {
            for (int col = 0; col < MATRIX_DIM; ++col) {
                resultFile << productMatrix[row][col] << "\t";
            }
            resultFile << std::endl;
        }
        resultFile << "Execution time: " << timeTaken.count() << " microseconds";
        resultFile.close();
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}