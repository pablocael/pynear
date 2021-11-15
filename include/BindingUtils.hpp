#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class BindingUtils {

public:
    template <class T>
    static py::array_t<T> bufferToNumpy1d(T* buffer, int dim) {
        py::capsule free_when_done(buffer, [](void *rawbuff) {
                T* buff = reinterpret_cast<T*>(rawbuff);
                delete[] buff;
        });

        return py::array_t<T>(
                {dim}, // shape
                {sizeof(T)}, // C-style contiguous strides for double
                buffer,
                free_when_done
            );
    }

    template <class T>
    static py::array_t<T> bufferToNumpy2d(T* buffer, int dim0, int dim1) {
        py::capsule free_when_done(buffer, [](void *rawbuff) {
                T* buff = reinterpret_cast<T*>(rawbuff);
                delete[] buff;
        });

        return py::array_t<T>(
                {dim0, dim1}, // shape
                {dim1*sizeof(T), sizeof(T)}, // C-style contiguous strides for double
                buffer,
                free_when_done
            );
    }

    template <class T>
    static py::array_t<T> bufferToNumpy3d(T* buffer, int dim0, int dim1, int dim2) {
        py::capsule free_when_done(buffer, [](void *rawbuff) {
                T* buff = reinterpret_cast<T*>(rawbuff);
                delete[] buff;
        });

        return py::array_t<T>(
                {dim0, dim1, dim2}, // shape
                {dim1 * dim2*sizeof(T), dim2 * sizeof(T), sizeof(T)}, // C-style contiguous strides for double
                buffer,
                free_when_done
            );
    }
};
