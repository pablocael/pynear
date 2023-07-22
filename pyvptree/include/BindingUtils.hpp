/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <type_traits>
#include <vector>

namespace py = pybind11;

class BindingUtils {

    public:
    template <class T, int... Dims> static py::array_t<T> bufferToNumpyNdArray(T *buffer) {
        /*
         *  :param buffer: this buffer will be destroyed automatically so this
         *  function will take over the ownership.
         */

        py::capsule free_when_done(buffer, [](void *rawbuff) {
            T *buff = reinterpret_cast<T *>(rawbuff);
            delete[] buff;
        });

        const int numDims = sizeof...(Dims);
        const int dims[] = {Dims...};

        int totalSize = 0;
        std::vector<int> offsets(numDims);
        for (int i = 0; i < numDims; ++i) {
            totalSize *= dims[i];
        }

        int offset = 1;
        for (int i = 0; i < numDims; ++i) {
            offset *= dims[i];
            offsets[i] = sizeof(T) * totalSize / offset;
        }

        return py::array_t<T>({Dims...},          // shape
                              std::move(offsets), // C-style contiguous strides for double
                              buffer, free_when_done);
    }
};
