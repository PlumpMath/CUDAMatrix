// Author: William Killian
// Contact: william.killian@gmail.com

// License:
//
//            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
//                    Version 2, December 2004
//
// Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>
//
// Everyone is permitted to copy and distribute verbatim or modified
// copies of this license document, and changing it is allowed as long
// as the name is changed.
//
//            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
//   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION
//
//  0. You just DO WHAT THE FUCK YOU WANT TO.
//

#ifndef GPU_MATRIX_HPP_
#define GPU_MATRIX_HPP_

#include <cstring>

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define CUDA_DEVICE_INLINE __device__ __forceinline__
#else
#define CUDA_HOST_DEVICE
#define CUDA_HOST_DEVICE_INLINE
#define CUDA_DEVICE_INLINE
#endif

enum MatrixAllocationType {
  Empty, CPU, GPU, Managed
};

template <typename T>
struct MatrixType {

  using value_type = T;
  T* __restrict__ raw;
  int col;
  int row;
  MatrixAllocationType allocationType;

  CUDA_HOST_DEVICE_INLINE
  int rows() const { return row; }

  CUDA_HOST_DEVICE_INLINE
  int cols() const { return col; }

  CUDA_HOST_DEVICE_INLINE
  int size() const { return row * col; }

  CUDA_HOST_DEVICE_INLINE
  T* data() { return raw; }

  CUDA_HOST_DEVICE_INLINE
  const T* data() const { return raw; }

  CUDA_HOST_DEVICE_INLINE
  T& operator() (int c, int r) {
    return raw [r * col + c];
  }

  CUDA_HOST_DEVICE_INLINE
  const T operator() (int c, int r) const {
    return raw [r * col + c];
  }

  CUDA_HOST_DEVICE_INLINE
  const T* cbegin() const { return raw; }

  CUDA_HOST_DEVICE_INLINE
  const T* begin() const { return raw; }

  CUDA_HOST_DEVICE_INLINE
  T* begin() { return raw; }

  CUDA_HOST_DEVICE_INLINE
  const T* cend() const { return raw + (col * row); }

  CUDA_HOST_DEVICE_INLINE
  const T* end() const { return raw + (col * row); }

  CUDA_HOST_DEVICE_INLINE
  T* end() { return raw + (col * row); }

  CUDA_HOST_DEVICE_INLINE
  T& operator[] (int i) { return raw[i]; }

  CUDA_HOST_DEVICE_INLINE
  const T operator[] (int i) const { return raw[i]; }

  static MatrixType<T> CreateCPU (int cols, int rows) {
    T* data = new T [rows * cols];
    return { data, cols, rows, MatrixAllocationType::CPU };
  }

  static MatrixType<T> CreateGPU (int cols, int rows) {
    T* data;
    cudaMalloc ((void**)(&data), sizeof(T) * rows * cols);
    return { data, cols, rows, MatrixAllocationType::GPU };
  }

  static MatrixType<T> CreateManaged (int cols, int rows) {
    cudaMallocManaged ((void**)(&data), sizeof(T) * rows * cols);
    return { data, cols, rows, MatrixAllocationType::Managed };
  }

  void copyFrom (const MatrixType<T>& other) {
    switch (allocationType) {
    case MatrixAllocationType::CPU:
      switch (other.allocationType) {
      case MatrixAllocationType::CPU:
      case MatrixAllocationType::Managed:
        ::memcpy (raw, other.raw, sizeof(T) * size());
        break;
      case MatrixAllocationType::GPU:
        cudaMemcpy (raw, other.raw, sizeof(T) * size(), cudaMemcpyDeviceToHost);
        break;
      default:
        break;
      }
      break;
    case MatrixAllocationType::GPU:
      switch (other.allocationType) {
      case MatrixAllocationType::CPU:
      case MatrixAllocationType::Managed:
        cudaMemcpy (raw, other.raw, sizeof(T) * size(), cudaMemcpyHostToDevice);
        break;
      case MatrixAllocationType::GPU:
        cudaMemcpy (raw, other.raw, sizeof(T) * size(), cudaMemcpyDeviceToDevice);
        break;
      default:
        break;
      }
      break;
    case MatrixAllocationType::Managed:
      switch (other.allocationType) {
      case MatrixAllocationType::CPU:
      case MatrixAllocationType::Managed:
        ::memcpy (raw, other.raw, sizeof(T) * size());
        break;
      case MatrixAllocationType::GPU:
        cudaMemcpy (raw, other.raw, sizeof(T) * size(), cudaMemcpyDeviceToHost);
        break;
      default:
        break;
      }
      break;
    default:
      break;
    }
  }

};

template <typename T>
MatrixType<T> CreateCPUMatrix (int cols, int rows) {
  return MatrixType<T>::CreateCPU (cols, rows);
}

template <typename T>
MatrixType<T> CreateGPUMatrix (int cols, int rows) {
  return MatrixType<T>::CreateGPU (cols, rows);
}

template <typename T>
MatrixType<T> CreateManagedMatrix (int cols, int rows) {
  return MatrixType<T>::CreateManaged (cols, rows);
}

template <typename T>
MatrixType<T> CreateCPUMatrix (const MatrixType<T>& other) {
  if (other.allocationType == MatrixAllocationType::Empty)
    return MatrixType<T> { nullptr, 0, 0, MatrixAllocationType::Empty };
  auto result = CreateCPUMatrix<T> (other.cols(), other.rows());
  result.copyFrom (other);
  return result;
}

template <typename T>
MatrixType<T> CreateGPUMatrix (const MatrixType<T>& other) {
  if (other.allocationType == MatrixAllocationType::Empty)
    return MatrixType<T> { nullptr, 0, 0, MatrixAllocationType::Empty };
  auto result = CreateGPUMatrix<T> (other.cols(), other.rows());
  result.copyFrom (other);
  return result;
}

template <typename T>
MatrixType<T> CreateManagedMatrix (const MatrixType<T>& other) {
  if (other.allocationType == MatrixAllocationType::Empty)
    return MatrixType<T> { nullptr, 0, 0, MatrixAllocationType::Empty };
  auto result = CreateManagedMatrix<T> (other.cols(), other.rows());
  result.copyFrom (other);
  return result;
}

template <typename T>
void FreeMatrix (MatrixType<T> & mat) {
  switch (mat.allocationType) {
  case CPU:
    delete[] mat.raw;
    break;
  case GPU:
  case Managed:
    cudaFree (mat.raw);
    break;
  default:
    break;
  }
  mat.raw = nullptr;
  mat.allocationType = MatrixAllocationType::Empty;
}

#endif
