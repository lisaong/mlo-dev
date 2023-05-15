#pragma once

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

cudaError_t CHECK(cudaError_t res) {
  if (cudaSuccess != res) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(res) << std::endl;
    exit(EXIT_FAILURE);
  }
  return res;
}

class Matrix {

public:
  Matrix(int rows, int columns) : _rows(rows), _columns(columns) {}

  int rows() const { return _rows; }
  int columns() const { return _columns; }
  uint64_t numElements() const { return _rows * _columns; }
  uint64_t size() const { return numElements() * sizeof(float); }

  void init(float value) {
    for (uint64_t i = 0; i < numElements(); ++i) {
      data[i] = value;
    }
  }

  void randomInit(int seed) {
    srand(seed);
    for (uint64_t i = 0; i < numElements(); ++i) {
      data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  bool operator==(const Matrix &other) const {
    if (_rows != other._rows || _columns != other._columns) {
      return false;
    }
    for (uint64_t i = 0; i < _rows * _columns; ++i) {
      if (std::fabs(data[i] - other.data[i]) >
          std::numeric_limits<float>::epsilon()) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Matrix &other) const { return !(*this == other); }

  void print() const {
    for (int i = 0; i < _rows; ++i) {
      for (int j = 0; j < _columns; ++j) {
        std::cout << data[i * _columns + j] << " ";
      }
      std::cout << std::endl;
    }
  }

  float *data;

protected:
  int _rows;
  int _columns;
};

class BandedMatrix : public Matrix {

public:
  BandedMatrix(int rows, int columns, int band)
      : Matrix(rows, band), _band(band), _expandedColumns(columns) {}
  int columns() const { return _expandedColumns; }
  int band() const { return _band; }

  void init(float value) {
    for (int i = 0; i < rows(); ++i) {
      for (int j = 0; j < _band; ++j) {
        if ((i + j) < columns()) {
          data[i * _band + j] = value;
        } else {
          // zero out the lower right triangle
          data[i * _band + j] = 0;
        }
      }
    }
  }

protected:
  int _band;
  int _expandedColumns;
};

class TransposedBandedMatrix : public Matrix {

public:
  TransposedBandedMatrix(int rows, int columns, int band)
      : Matrix(band, columns), _band(band), _expandedRows(rows) {}

  int rows() const { return _expandedRows; }
  int band() const { return _band; }

  void init(float value) {
    for (int i = 0; i < _band; ++i) {
      for (int j = 0; j < columns(); ++j) {
        if ((i + j) < rows()) {
          data[i * columns() + j] = value;
        } else {
          // zero out the lower right triangle
          data[i * columns() + j] = 0;
        }
      }
    }
  }

protected:
  int _band;
  int _expandedRows;
};

void bandedMatMul_CPU(int n0, int n1, int n2, float *t0, const float *t1,
                      const float *t2) {
  /*
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                t0[i, j] += t1[i, k] * t2[i + k, j]
  */
  int i, j, k;
  for (i = 0; i < n0; ++i) {
    for (j = 0; j < n1; ++j) {
      for (k = 0; k < n2 && (i + k) < n0; ++k) {
        t0[i * n1 + j] += t1[i * n2 + k] * t2[(i + k) * n1 + j];
      }
    }
  }
}

bool checkCorrectness(int n0, int n1, int n2, const Matrix &T0,
                      const BandedMatrix &T1, const Matrix &T2) {
  Matrix T0_CPU(T0.rows(), T0.columns());

  T0_CPU.data = reinterpret_cast<float *>(malloc(T0_CPU.size()));
  T0_CPU.init(11.0f);

  bandedMatMul_CPU(n0, n1, n2, T0_CPU.data, T1.data, T2.data);

#if DEBUG
  std::cout << "T0_CPU: " << std::endl;
  T0_CPU.print();
  std::cout << "T0: " << std::endl;
  T0.print();
#endif // DEBUG

  bool result = T0_CPU == T0;
  if (result) {
    std::cout << "Values match" << std::endl;
  } else {
    std::cerr << "Values do not match" << std::endl;
  }

  free(T0_CPU.data);
  return result;
}

bool checkCorrectness(int n0, int n1, int n2, const Matrix &T0,
                      const BandedMatrix &T1,
                      const TransposedBandedMatrix &T2) {
  Matrix T2_CPU(T2.rows(), T2.columns());

  T2_CPU.data = reinterpret_cast<float *>(malloc(T2_CPU.size()));
  T2_CPU.init(33.0f);

  bool result = checkCorrectness(n0, n1, n2, T0, T1, T2_CPU);

  free(T2_CPU.data);
  return result;
}