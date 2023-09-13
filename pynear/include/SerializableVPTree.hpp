
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
        if (this->_rootPartition == nullptr) {
            return state;
        }

        // Create a writer that will write to the state object
        SerializedStateObjectWriter writer(state);
        writer.writeUserVector<T, serializer>(this->_examples);
        writer.writeVector<int64_t>(this->_indices);

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
        this->_indices = reader.readVector<int64_t>();

        // Deserialize partitions
        deserializeLevelPartitions(reader);
    };

    void serializeLevelPartitions(SerializedStateObjectWriter &writer) const {

        std::vector<const VPLevelPartition<distance_type> *> flattenTreeState;
        flattenTreePartitions(this->_rootPartition, flattenTreeState);

        // reverse the tree state since we will write it in a stack for serializing
        for (const VPLevelPartition<distance_type> *elem : flattenTreeState) {
            if (elem == nullptr) {
                writer.write((float)(0));
                writer.write((int64_t)(-1));
                writer.write((int64_t)(-1));
                continue;
            }

            writer.write((float)(elem->radius()));
            writer.write((int64_t)(elem->start()));
            writer.write((int64_t)(elem->end()));
        }
    }

    void deserializeLevelPartitions(SerializedStateObjectReader &reader) { this->_rootPartition = rebuildFromState(reader); }

    void flattenTreePartitions(const VPLevelPartition<distance_type> *root,
                               std::vector<const VPLevelPartition<distance_type> *> &flattenTreeState) const {
        // visit partitions tree in preorder write all values.
        // implement pre order using a vector as a stack
        flattenTreeState.push_back(root);
        if (root != nullptr) {
            flattenTreePartitions(root->left(), flattenTreeState);
            flattenTreePartitions(root->right(), flattenTreeState);
        }
    }

    VPLevelPartition<distance_type> *rebuildFromState(SerializedStateObjectReader &reader) {
        if (reader.isEmpty()) {
            return nullptr;
        }

        float radius = reader.read<float>();
        int64_t indexStart = reader.read<int64_t>();
        int64_t indexEnd = reader.read<int64_t>();
        if (indexEnd == -1) {
            return nullptr;
        }

        VPLevelPartition<distance_type> *root = new VPLevelPartition<distance_type>(radius, indexStart, indexEnd);
        VPLevelPartition<distance_type> *left = rebuildFromState(reader);
        VPLevelPartition<distance_type> *right = rebuildFromState(reader);
        root->setChild(left, right);
        return root;
    }
};

} // namespace vptree
