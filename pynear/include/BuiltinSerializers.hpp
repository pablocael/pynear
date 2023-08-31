#pragma once

#include <vector>
#include <Eigen/Core>

namespace vptree {

// Built in serializer for vector of vectors
template <typename T> void ndarraySerializer(const std::vector<std::vector<T>> &input, std::vector<uint8_t> &output) {
    /* 
     * This serializer function should write state from lower to higher addresses.
     * It writes a ndarray like vector (e.g: a vector of vectors) to a byte array.
     * It will serialize meta information about the vector such as its size and dimension.
     * The final format will be:
     * [totalSize(size_t)][dimension(size_t)][element1(std::vector<T>)]...[elementN(std::vector<T>)]
     *
     * See SerializableVPTree declaration for more information.
    */
    auto totalSize = input.size();
    if (totalSize == 0) {
        return;
    }

    size_t dimension = input[0].size();
    auto totalBytes = totalSize * dimension * sizeof(T);
    uint8_t *data = output.data();
    // add space for all elements of given dimensions + total size + dimension in bytes
    output.resize(output.size() + totalBytes + 2 * sizeof(size_t));

    // store total size and dimension first
    (*(size_t *)(data)) = totalSize;
    data += sizeof(size_t);
    (*(size_t *)(data)) = dimension;
    data += sizeof(size_t);
    
    // store elements
    for (const auto &element : input) {
        std::memcpy(data, &element.front(), dimension * sizeof(T));
        data += dimension * sizeof(T);
    }
};

// Built in deserializer for vector of vectors
template <typename T> std::vector<std::vector<T>> ndarrayDeserializer(const uint8_t *input, size_t &readBytes) {
    /*
     * This deserializer function should read state from lower to higher addresses.
     * See ndarraySerializer for more information.
     */

    uint8_t *data = const_cast<uint8_t *>(input);
    // input points to after the data block, first must decrement

    // read total size and dimension first
    size_t totalSize = (*(size_t *)(data));
    data += sizeof(size_t);
    size_t dimension = (*(size_t *)(data));
    data += sizeof(size_t);

    size_t elementSize = dimension * sizeof(T);
    size_t totalBytes = totalSize * elementSize;

    std::vector<std::vector<T>> result;
    result.resize(totalSize);
    for (auto &element : result) {
        element.resize(dimension);
        std::memcpy(&element.front(), data, elementSize);
        data += elementSize;
    }

    readBytes = totalBytes + 2 * sizeof(size_t);

    return result;
};

} // namespace vptree;
