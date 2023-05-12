#pragma once

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

class Matrix {

public:
  Matrix(int rows, int columns) : _rows(rows), _columns(columns) {}

  int rows() { return _rows; }
  int columns() { return _columns; }
  uint64_t numElements() { return _rows * _columns; }
  uint64_t size() { return numElements() * sizeof(*data); }
  void init(float value) {
    for (uint64_t i = 0; i < numElements(); ++i) {
      data[i] = value;
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

  float *data;

protected:
  int _rows;
  int _columns;
};

class BandedMatrix : public Matrix {

public:
  BandedMatrix(int rows, int columns) : Matrix(rows, columns) {}
  int columns() { return _rows + _columns - 1; }
  int diagonals() { return _columns; }
};

cudaError_t CHECK(cudaError_t res) {
  if (cudaSuccess != res) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(res) << std::endl;
    exit(EXIT_FAILURE);
  }
  return res;
}
