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
      : Matrix(rows, band), _band(band), _expandedColumns(columns) {}
  int columns() const { return _expandedColumns; }

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