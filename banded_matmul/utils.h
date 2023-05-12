#pragma once

#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>

class Matrix {

public:
  Matrix(int w, int h) : _w(w), _h(h) {}

  int width() { return _w; }
  int height() { return _h; }
  uint64_t numElements() { return _w * _h; }
  uint64_t size() { return numElements() * sizeof(*data); }
  void init(float value) {
    for (auto i = 0; i < numElements(); ++i) {
      data[i] = value;
    }
  }

  float *data;

protected:
  int _w;
  int _h;
};

class BandedMatrix : public Matrix {

public:
  BandedMatrix(int w, int h) : Matrix(w, h) {}
  int width() { return _w + _h - 1; }
};

cudaError_t CHECK(cudaError_t res) {
  if (cudaSuccess != res) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(res) << std::endl;
    exit(EXIT_FAILURE);
  }
  return res;
}
