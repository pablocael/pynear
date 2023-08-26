
/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#pragma once

#include "ISerializable.hpp"
#include "VPTree.hpp"

namespace vptree {

template <typename T, typename distance_type, distance_type (*distance)(const T &, const T &), void (*serializer)(const T &, std::vector<uint8_t> &),
          T (*deserializer)(const std::vector<uint8_t> &)>
class SerializableVPTree : public VPTree<T, distance_type, distance>, public ISerializable {
    public:

    SerializableVPTree() = default;
    SerializableVPTree(const SerializableVPTree<T, distance_type, distance, serializer, deserializer> &other) {
        this->_examples = other._examples;
        this->_indices = other._indices;
        if(this->_rootPartition != nullptr) {
            this->_rootPartition = other._rootPartition->deepcopy();
        }
    }

    const SerializableVPTree<T, distance_type, distance, serializer, deserializer> &
    operator=(const SerializableVPTree<T, distance_type, distance, serializer, deserializer> &other) {
        this->_examples = other._examples;
        this->_indices = other._indices;
        if(this->_rootPartition != nullptr) {
            this->_rootPartition = other._rootPartition->deepcopy();
        }

        return *this;
    }

    SerializedState serialize() const override {
        if (this->_rootPartition == nullptr) {
            return SerializedState();
        }

        size_t element_size = 0;
        size_t num_elements_per_example = 0;
        size_t total_size = (size_t)(3 * sizeof(size_t));

        if (this->_examples.size() > 0) {

            // total size is the _examples total size + the examples array size plus element size
            // _examples[0] is an array of some type (variable)
            num_elements_per_example = this->_examples[0].size();
            element_size = sizeof(this->_examples[0][0]);
            int64_t total_elements_size = num_elements_per_example * element_size;
            total_size += this->_examples.size() * (sizeof(int64_t) + total_elements_size);
        }

        SerializedState state;
        state.reserve(total_size);

        /* for (const VPTreeElement &elem : _examples) { */
        /*     state.push(elem.originalIndex); */

        /*     for (size_t i = 0; i < num_elements_per_example; ++i) { */
        /*         // since we dont know the sub element type of T (T is an array of something) */
        /*         // we need to copy using memcopy and memory size */
        /*         state.push_by_size(&elem[i], element_size); */
        /*     } */
        /* } */

        state.push(this->_examples.size());
        state.push(num_elements_per_example);
        state.push(element_size);

        if (state.size() != total_size) {
            throw new std::out_of_range("invalid serialization state, offsets dont match!");
        }

        SerializedState partition_state = this->_rootPartition->serialize();
        partition_state += state;
        partition_state.buildChecksum();
        return partition_state;
    }

    void deserialize(const SerializedState &state) override {
        this->clear();
        if (state.data.empty()) {
            return;
        }

        if (!state.isValid()) {
            throw new std::invalid_argument("invalid state - checksum mismatch");
        }

        SerializedState copy(state);

        this->_rootPartition = new VPLevelPartition<distance_type>();

        size_t elem_size = copy.pop<size_t>();
        size_t num_elements_per_example = copy.pop<size_t>();
        size_t num_examples = copy.pop<size_t>();

        /* _examples.reserve(num_examples); */
        /* _examples.resize(num_examples); */
        /* for (int64_t i = num_examples - 1; i >= 0; --i) { */

        /*     auto &example = _examples[i]; */
        /*     auto &val = example; */
        /*     val.resize(num_elements_per_example); */

        /*     for (int64_t j = num_elements_per_example - 1; j >= 0; --j) { */
        /*         copy.pop_by_size(&val[j], elem_size); */
        /*     } */
        /*     int64_t originalIndex = copy.pop<int64_t>(); */
        /*     example.originalIndex = originalIndex; */
        /* } */

        this->_rootPartition->deserialize(copy);
    };
};

} // namespace vptree
