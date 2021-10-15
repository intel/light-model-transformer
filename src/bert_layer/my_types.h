// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __HPJ_MATRIX_H_
#define __HPJ_MATRIX_H_
#include <assert.h>

namespace hpj {

template <typename T>
class Matrix {
private:
    // A sub matrix of others, if true
    bool shadow_;
    int rows_;
    int cols_;
    int stride_;
    T* data_;
    // How many elements was allocated
    int alloc_size_;

    Matrix& operator=(const Matrix &m);

public:
    Matrix() {
        this->shadow_ = false;
        this->rows_ = 0;
        this->cols_ = 0;
        this->stride_ = 0;
        this->data_ = NULL;
        this->alloc_size_ = 0;
    }

    Matrix(Matrix &m, int start_row, int rows, int start_col, int cols) {
        this->shadow_ = true;
        this->rows_ = rows;
        this->cols_ = cols;
        this->stride_ = m.stride_;
        this->data_ = m.data_ + start_row * m.stride_ + start_col;
        this->alloc_size_ = 0;
    }

    Matrix(Matrix &m) {
        this->shadow_ = true;
        this->rows_ = m.rows_;
        this->cols_ = m.cols_;
        this->stride_ = m.stride_;
        this->data_ = m.data_;
        this->alloc_size_ = 0;
    }

    // Create dilated matrix, for example, if dilation = 2, then select the 1st, 3rd, 5th, ... lines
    Matrix(Matrix &m, int start_row, int dilation, bool unused) {
        this->shadow_ = true;
        this->rows_ = m.rows_ / dilation;
        this->cols_ = m.cols_;
        this->stride_ = m.stride_ * dilation;
        this->data_ = m.data_ + start_row * m.stride_;
        this->alloc_size_ = 0;
    }

    Matrix(Matrix &m, int start_row, int rows) {
        this->shadow_ = true;
        this->rows_ = rows;
        this->cols_ = m.cols_;
        this->stride_ = m.stride_;
        this->data_ = m.data_ + start_row * m.stride_;
        this->alloc_size_ = 0;
    }

    Matrix(T *data, int rows, int cols, int stride) {
        this->shadow_ = true;
        this->rows_ = rows;
        this->cols_ = cols;
        this->stride_ = stride;
        this->data_ = data;
        this->alloc_size_ = 0;
    }

    ~Matrix() {
        this->Release();
    }

    void Resize(int rows, int cols) {
        assert(!shadow_);

        if (rows == rows_ && cols == cols_) {
            return;
        }
        if (rows < 0 && cols < 0) {
            return;
        }
        if (rows == 0 || cols == 0) {
            this->Release();
            return;
        }
        if (cols > 16) {
            //int skip = (16 - cols % 16) % 16;
            //stride_ = cols + skip;
            //if (stride_ % 256 == 0) {
            //    stride_ += 4;
            //}
            stride_ = cols;
        } else { // for narrow matrix, not padding any more
            stride_ = cols;
        }
        rows_ = rows;
        cols_ = cols;
        if (alloc_size_ >= stride_ * rows) {
            return;
        } else {
            if (data_) {
                free(data_);
            }
            alloc_size_ = stride_ * rows_;
            data_ = (T *)aligned_alloc(64, sizeof(T) * alloc_size_);
            if (data_ == NULL) {
                throw std::bad_alloc();
            }
        }
    }
    T* Data() {
        return data_;
    }
    const T* Data() const {
        return data_;
    }
    void Release() {
        if (!shadow_ && data_) {
            free(data_);
            data_ = NULL;
        }
        rows_ = 0;
        cols_ = 0;
        stride_ = 0;
        alloc_size_ = 0;
    }
    int Rows() const {
        return rows_;
    }
    int Cols() const {
        return cols_;
    }
    int Stride() const {
        return stride_;
    }
    T* Row(const int idx) {
        //assert(idx < rows_ && idx >= 0);
        return data_ + stride_ * idx;
    }
    const T* Row(const int idx) const {
        return data_ + stride_ * idx;
    }
    T& operator()(int r, int c) { 
        //assert(r >= 0 && r < rows_ && c >= 0 && c < cols_);
        return *(data_ + r * stride_ + c );
    }
};

template <typename T>
class Vector {
private:
    T* data_;
    int size_;
    int alloc_size_;

public:
    Vector() {
        data_ = NULL;
        size_ = 0;
        alloc_size_ = 0;
    }
    ~Vector() {
        this->Release();
    }
    void Resize(int size) {
        if (size <= 0){
            this->Release();
            return;
        }
        int skip = (16 - size % 16) % 16;
        if (alloc_size_ >= size + skip) { // space is enough
            size_ = size;
            return;
        }

        alloc_size_ = size + skip;
        size_ = size;
        if (data_) {
            free(data_);
        }
        data_ = (T *)aligned_alloc(64, sizeof(T) * alloc_size_);
        if (data_ == NULL) {
            throw std::bad_alloc();
        }
    }
    void SetZero() {
        memset(data_, 0, sizeof(T) * size_);
    }
    T* Data() {
        return data_;
    }
    void Release() {
        if (data_) {
            free(data_);
            data_ = NULL;
        }
        size_ = 0;
        alloc_size_ = 0;
    }
    int Size() {
        return size_;
    }
}; 
} // end namespace

#endif
