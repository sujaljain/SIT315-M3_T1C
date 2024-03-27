#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>
#include <omp.h> // Include OpenMP library

using namespace std::chrono;

const int MATRIX_DIM = 200;

void fillMatrixWithRandomValues(int matrix[][MATRIX_DIM]) {
    for (int row = 0; row < MATRIX_DIM; ++row)
        for (int col = 0; col < MATRIX_DIM; ++col)
            matrix[row][col] = rand() % 100;
}

void calculateProductOfMatrices(const int firstMatrix[][MATRIX_DIM], const int secondMatrix[][MATRIX_DIM], int productMatrix[][MATRIX_DIM], int startRow, int endRow) {
    #pragma omp parallel for
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
    srand(time(nullptr)); // Seed the random number generator

    int firstMatrix[MATRIX_DIM][MATRIX_DIM];
    int secondMatrix[MATRIX_DIM][MATRIX_DIM];
    int productMatrix[MATRIX_DIM][MATRIX_DIM] = {0};

    fillMatrixWithRandomValues(firstMatrix);
    fillMatrixWithRandomValues(secondMatrix);

    auto startMoment = high_resolution_clock::now();

    // Calculate product of matrices using OpenMP
    calculateProductOfMatrices(firstMatrix, secondMatrix, productMatrix, 0, MATRIX_DIM);

    auto endMoment = high_resolution_clock::now();
    auto timeTaken = duration_cast<microseconds>(endMoment - startMoment);

    std::cout << "Multiplication completed in: " << timeTaken.count() << " microseconds" << std::endl;

    // Write the result matrix and execution time to a file
    std::ofstream resultFile("Result_matrix.txt");
    for (int row = 0; row < MATRIX_DIM; ++row) {
        for (int col = 0; col < MATRIX_DIM; ++col) {
            resultFile << productMatrix[row][col] << "\t";
        }
        resultFile << std::endl;
    }
    resultFile << "Execution time: " << timeTaken.count() << " microseconds";
    resultFile.close();

    return 0;
}
