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

  float *data;

protected:
  int _rows;
  int _columns;
};

class BandedMatrix : public Matrix {

public:
  BandedMatrix(int rows, int columns, int band)
      : Matrix(rows, band), _expandedColumns(columns) {}
  int columns() const { return _expandedColumns; }

protected:
  int _expandedColumns;
};

class TransposedBandedMatrix : public Matrix {

public:
  TransposedBandedMatrix(int rows, int columns, int band)
      : Matrix(band, columns), _expandedRows(rows) {}

  int rows() const { return _expandedRows; }

protected:
  int _expandedRows;
};