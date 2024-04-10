#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <omp.h>

using namespace std::chrono;

const int MATRIX_DIM = 300;
const int NUM_PROCESSES = 4;
const int NUM_THREADS = 2;

void fillMatrixWithRandomValues(int matrix[][MATRIX_DIM]) {
    for (int row = 0; row < MATRIX_DIM; ++row)
        for (int col = 0; col < MATRIX_DIM; ++col)
            matrix[row][col] = rand() % 100;
}

void calculateProductOfMatrices(const int firstMatrix[][MATRIX_DIM], const int secondMatrix[][MATRIX_DIM], int productMatrix[][MATRIX_DIM], int startRow, int endRow) {
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int row = startRow; row < endRow; ++row) {
        for (int col = 0; col < MATRIX_DIM; ++col) {
            productMatrix[row][col] = 0;
            for (int k = 0; k < MATRIX_DIM; ++k) {
                productMatrix[row][col] += firstMatrix[row][k] * secondMatrix[k][col];
            }
        }
    }
}

int main() {
    srand(time(nullptr));
    int firstMatrix[MATRIX_DIM][MATRIX_DIM];
    int secondMatrix[MATRIX_DIM][MATRIX_DIM];
    int productMatrix[MATRIX_DIM][MATRIX_DIM] = {0};

    fillMatrixWithRandomValues(firstMatrix);
    fillMatrixWithRandomValues(secondMatrix);

    auto startMoment = high_resolution_clock::now();

    // Calculate product of matrices using OpenMP
    int rowsPerProcess = MATRIX_DIM / NUM_PROCESSES;
    #pragma omp parallel for num_threads(NUM_PROCESSES)
    for (int i = 0; i < NUM_PROCESSES; ++i) {
        int startRow = i * rowsPerProcess;
        int endRow = (i + 1) * rowsPerProcess;
        calculateProductOfMatrices(firstMatrix, secondMatrix, productMatrix, startRow, endRow);
    }

    auto endMoment = high_resolution_clock::now();
    auto timeTaken = duration_cast<seconds>(endMoment - startMoment);

    std::cout << "MPI Matrix Multiplication Performance with OpenMP" << std::endl;
    std::cout << "Dimension: " << MATRIX_DIM << std::endl;
    std::cout << "Processes: " << NUM_PROCESSES << std::endl;
    std::cout << "Threads: " << NUM_THREADS << std::endl;
    std::cout << "Runtime: " << timeTaken << std::endl;

    return 0;
}
