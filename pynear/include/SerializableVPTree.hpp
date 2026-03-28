
/*
 *  MIT Licence
 *  Copyright 2021 Pablo Carneiro Elias
 */

#pragma once

#include "ISerializable.hpp"
#include "VPTree.hpp"

namespace vptree {

template <typename T, typename distance_type, distance_type (*distance)(const T &, const T &),
          void (*serializer)(const std::vector<T> &, std::vector<uint8_t> &), std::vector<T> (*deserializer)(const uint8_t *, size_t &)>
class SerializableVPTree : public VPTree<T, distance_type, distance>, public ISerializable {
    /*
     * SerializableVPTree class that works with custom serializer and deserializer functions as a template.
     * Since T is a generic user type, serialization need custom user functions to be able to read/write
     * custom user objects.
     *
     * Template arguments:
     * - T: a custom user type that will compose the VPTree element. Distance function must input this type.
     * - distance_type: the numeric type for the distance value retrieved by distance function when
     * - distance: a function pointer of a distance operator that will measure the distance between two objects
     *   of type T.
     *   measuring ddistances between two objects of type T.
     * - serializer: a function serializes and array of user type T from a bytearray.
     *      void serialized(const std::vector<T>& input, std::vector<uint8_t>& output);
     *      Users must write data at the end of the output vector in serializer function (append data).
     * - deserializer function deserializes a vector of custom user type T from a bytearray:
     *      std::vector<T> deserialize(const uint8_t* input, size_t& readBytes);
     *      The deserializer must also return how many bytes were read from input buffer in readBytes variable. Also,
     *      the deserializer function input pointer points to start of the data block to be read, so user needs to first
     *      to read in same order they wrote the data (as in a file descriptor).
     */
public:
    SerializableVPTree() : VPTree<T, distance_type, distance>(){};
    SerializableVPTree(std::vector<T> &&examples) : VPTree<T, distance_type, distance>(examples) {}
    SerializableVPTree(const std::vector<T> &examples) : VPTree<T, distance_type, distance>(examples) {}
    SerializableVPTree(const SerializableVPTree<T, distance_type, distance, serializer, deserializer> &other)
        : VPTree<T, distance_type, distance>(other) {}

    const SerializableVPTree<T, distance_type, distance, serializer, deserializer> &
    operator=(const SerializableVPTree<T, distance_type, distance, serializer, deserializer> &other) {
        VPTree<T, distance_type, distance>::operator=(other);
        return *this;
    }

    SerializedStateObject serialize() const override {

        SerializedStateObject state;
        if (this->_rootIdx == -1) {
            return state;
        }

        // Create a writer that will write to the state object
        SerializedStateObjectWriter writer(state);
        writer.writeUserVector<T, serializer>(this->_examples);
        writer.writeVector<int32_t>(this->_indices);

        // Serialize partitions
        serializeLevelPartitions(writer);

        return state;
    }

    void deserialize(const SerializedStateObject &state) override {

        this->clear();
        if (state.isEmpty()) {
            return;
        }

        if (!state.isValid()) {
            throw new std::invalid_argument("invalid state - checksum mismatch");
        }

        SerializedStateObjectReader reader(state);
        this->_examples = reader.readUserVector<T, deserializer>();
        this->_indices = reader.readVector<int32_t>();

        // Deserialize partitions
        deserializeLevelPartitions(reader);
    };

    void serializeLevelPartitions(SerializedStateObjectWriter &writer) const {

        // Flatten the tree using indices into the pool via preorder traversal
        std::vector<int32_t> stack;
        // We'll do a preorder traversal using the pool and indices
        // We serialize: for each node visited in preorder, write radius/start/end
        // For null children, we write a sentinel (-1 indexEnd)
        flattenAndWritePartitions(writer, this->_rootIdx);
    }

    void flattenAndWritePartitions(SerializedStateObjectWriter &writer, int32_t idx) const {
        if (idx < 0) {
            // write sentinel null node
            writer.write((float)(0));
            writer.write((int32_t)(-1));
            writer.write((int32_t)(-1));
            return;
        }

        const VPLevelPartition<distance_type> &node = this->_nodePool[idx];
        writer.write((float)(node.radius()));
        writer.write((int32_t)(node.start()));
        writer.write((int32_t)(node.end()));

        flattenAndWritePartitions(writer, node.left_idx());
        flattenAndWritePartitions(writer, node.right_idx());
    }

    void deserializeLevelPartitions(SerializedStateObjectReader &reader) {
        this->_nodePool.clear();
        this->_rootIdx = rebuildFromState(reader);
    }

    int32_t rebuildFromState(SerializedStateObjectReader &reader) {
        if (reader.isEmpty()) {
            return -1;
        }

        float radius = reader.read<float>();
        int32_t indexStart = reader.read<int32_t>();
        int32_t indexEnd = reader.read<int32_t>();
        if (indexEnd == -1) {
            return -1;
        }

        // Push node to pool, get its index
        this->_nodePool.push_back(VPLevelPartition<distance_type>(radius, indexStart, indexEnd));
        int32_t nodeIdx = static_cast<int32_t>(this->_nodePool.size() - 1);

        // IMPORTANT: rebuild children BEFORE using nodeIdx to index into pool,
        // because push_back may reallocate. We save idx and re-index after.
        int32_t left_idx = rebuildFromState(reader);
        int32_t right_idx = rebuildFromState(reader);

        // After recursion, re-access by saved index (pool may have reallocated)
        this->_nodePool[nodeIdx].setChildIdx(left_idx, right_idx);

        return nodeIdx;
    }
};

} // namespace vptree
