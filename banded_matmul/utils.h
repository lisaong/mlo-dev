#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <mma.h>

void CHECK(cudaError_t res) {
  if (cudaSuccess != res) {
    throw std::runtime_error(std::string(cudaGetErrorString(res)));
  }
}

template <typename T>
__global__ void initWith(T num, T *a, int rows, int columns) {

  int i, j;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < columns;
         j += blockDim.y * gridDim.y) {
      a[i * columns + j] = num;
    }
  }
}

template <typename T>
__global__ void initBandedWith(T num, T *a, int rows, int columns, int band) {

  int i, j;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < band;
         j += blockDim.y * gridDim.y) {

      if ((i + j) < columns) {
        a[i * band + j] = num;
      } else {
        // zero out the lower right triangle
        a[i * band + j] = 0;
      }
    }
  }
}

template <typename T> class Matrix {

public:
  Matrix(int rows, int columns, bool columnMajor = false)
      : _rows(rows), _columns(columns), _columnMajor(columnMajor) {}

  int rows() const { return _rows; }
  int columns() const { return _columns; }
  uint64_t numElements() const { return _rows * _columns; }
  uint64_t size() const { return numElements() * sizeof(T); }
  bool columnMajor() const { return _columnMajor; }

  void init(T value) {
    for (uint64_t i = 0; i < numElements(); ++i) {
      data[i] = value;
    }
  }

  void randomInit(int seed) {
    srand(seed);
    for (uint64_t i = 0; i < _rows; ++i) {
      for (uint64_t j = 0; j < _columns; ++j) {
        if (_columnMajor)
          data[j * _rows + i] = static_cast<T>(rand()) / RAND_MAX;
        else
          data[i * _columns + j] = static_cast<T>(rand()) / RAND_MAX;
      }
    }
  }

  bool operator==(const Matrix &other) const {
    if (_rows != other._rows || _columns != other._columns) {
      return false;
    }
    for (uint64_t i = 0; i < _rows * _columns; ++i) {
      if (std::fabs(data[i] - other.data[i]) > kEpsilon) {
#if DEBUG
        std::cout << "Mismatch at " << i << ": " << data[i] << " "
                  << other.data[i]
                  << ", absolute diff: " << std::fabs(data[i] - other.data[i])
                  << std::endl;
#endif
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Matrix &other) const { return !(*this == other); }

  void print(int maxDim = 0) const {
    int rows = maxDim == 0 ? _rows : maxDim;
    int columns = maxDim == 0 ? _columns : maxDim;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < columns; ++j) {
        if (_columnMajor)
          std::cout << data[j * _rows + i] << " ";
        else
          std::cout << data[i + _columns * j] << " ";
      }
      std::cout << std::endl;
    }
  }

  T *data;

protected:
  int _rows;
  int _columns;
  bool _columnMajor;
};

template <typename T> class BandedMatrix : public Matrix<T> {

public:
  BandedMatrix(int rows, int band) : Matrix<T>(rows, band), _band(band) {}
  int columns() const { return this->_rows + _band; }
  int band() const { return _band; }

  // The lower right triangle of a banded matrix is typcially padded with zeros.
  // [ x x x 0 0 0 0 0 ]      [ x x x ]
  // [ 0 x x x 0 0 0 0 ]      [ x x x ]
  // [ 0 0 x x x 0 0 0 ]      [ x x x ]
  // [ 0 0 0 x x x 0 0 ]  =>  [ x x x ]
  // [ 0 0 0 0 x x x 0 ]      [ x x x ]
  // [ 0 0 0 0 0 x x x ]      [ x x x ]
  // [ 0 0 0 0 0 0 x x ]      [ x x 0 ]  A[6, 2] = 0
  // [ 0 0 0 0 0 0 0 x ]      [ x 0 0 ]  A[7, 1] = A[7, 2] = 0

  void init(float value) {
    for (int i = 0; i < this->rows(); ++i) {
      for (int j = 0; j < _band; ++j) {
        if ((i + j) < columns()) {
          this->data[i * _band + j] = value;
        } else {
          // zero out the lower right triangle
          this->data[i * _band + j] = 0;
        }
      }
    }
  }

protected:
  int _band;
};

void bandedMatMul_CPU(int n0, int n1, int n2, float *t0, const float *t1,
                      const float *t2, bool t2ColumnMajor = false) {
  /*
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                t0[i, j] += t1[i, k] * t2[i + k, j]
  */
  int i, j, k;
  for (i = 0; i < n0; ++i) {
    for (j = 0; j < n1; ++j) {
      for (k = 0; i + k < n2; ++k) {
        if (t2ColumnMajor)
          t0[i * n1 + j] += t1[i * n2 + k] * t2[(i + k) + j * n1];
        else
          t0[i * n1 + j] += t1[i * n2 + k] * t2[(i + k) * n1 + j];
      }
    }
  }
}

template <typename TIn, typename TOut>
void fillMatrices(Matrix<TOut> &T0, BandedMatrix<TIn> &T1, Matrix<TIn> &T2,
                  dim3 blocks, dim3 threads, int deviceId) {

  T0.randomInit(11);
  CHECK(cudaMemPrefetchAsync(T0.data, T0.size(), deviceId));
  initBandedWith<<<blocks, threads>>>(22.0f, T1.data, T1.rows(), T1.columns(),
                                      T1.band());
  T2.randomInit(123);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemPrefetchAsync(T2.data, T2.size(), deviceId));
}

template <typename TIn, typename TOut>
bool checkCorrectness(int n0, int n1, int n2, const Matrix<TOut> &T0,
                      const BandedMatrix<TIn> &T1, const Matrix<TIn> &T2) {
  Matrix<TOut> T0_CPU(T0.rows(), T0.columns());

  T0_CPU.data = reinterpret_cast<TOut *>(malloc(T0_CPU.size()));
  T0_CPU.randomInit(11);

  // bandedMatMul_CPU(n0, n1, n2, T0_CPU.data, T1.data, T2.data,
  // T2.columnMajor());

#if DEBUG
  std::cout << "T0_CPU: " << std::endl;
  T0_CPU.print(10);
  std::cout << "T0: " << std::endl;
  T0.print(10);
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

int ceildiv(int value, int divisor) { return (value + divisor - 1) / divisor; }