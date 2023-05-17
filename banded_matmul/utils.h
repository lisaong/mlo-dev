#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>

void CHECK(cudaError_t res) {
  if (cudaSuccess != res) {
    throw std::runtime_error(std::string(cudaGetErrorString(res)));
  }
}

__global__ void initWith(float num, float *a, int rows, int columns) {

  int i, j;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < columns;
         j += blockDim.y * gridDim.y) {
      a[i * columns + j] = num;
    }
  }
}

__global__ void initWith(half num, half *a, int rows, int columns) {

  int i, j;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < columns;
         j += blockDim.y * gridDim.y) {
      a[i * columns + j] = num;
    }
  }
}

__global__ void initBandedWith(float num, float *a, int rows, int columns,
                               int band) {

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

__global__ void initBandedWith(half num, half *a, int rows, int columns,
                               int band) {

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
  Matrix(int rows, int columns) : _rows(rows), _columns(columns) {}

  int rows() const { return _rows; }
  int columns() const { return _columns; }
  uint64_t numElements() const { return _rows * _columns; }
  uint64_t size() const { return numElements() * sizeof(T); }

  void init(T value) {
    for (uint64_t i = 0; i < numElements(); ++i) {
      data[i] = value;
    }
  }

  void randomInit(int seed) {
    srand(seed);
    for (uint64_t i = 0; i < numElements(); ++i) {
      data[i] = static_cast<T>(rand()) / RAND_MAX;
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

  T *data;

protected:
  int _rows;
  int _columns;
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

void bandedMatMul_CPU(int n0, int n1, int n2, float *t0, const half *t1,
                      const half *t2) {
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
        t0[i * n1 + j] += (float)t1[i * n2 + k] * (float)t2[(i + k) * n1 + j];
      }
    }
  }
}

template <typename TIn, typename TOut>
bool checkCorrectness(int n0, int n1, int n2, const Matrix<TOut> &T0,
                      const BandedMatrix<TIn> &T1, const Matrix<TIn> &T2) {
  Matrix<TOut> T0_CPU(T0.rows(), T0.columns());

  T0_CPU.data = reinterpret_cast<TOut *>(malloc(T0_CPU.size()));
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

int ceildiv(int value, int divisor) {
  if (value % divisor == 0) {
    return value / divisor;
  }
  return value / divisor + 1;
}