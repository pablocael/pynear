#pragma once

#include "crc32.hpp"
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

namespace vptree {

class SerializedStateObject {
    /*
     * Stores a serialized object in a byte array. The object can be read and written using
     * SerializedStateObjectReader and SerializedStateObjectWriter.
     */
public:
    friend class SerializedStateObjectReader;
    friend class SerializedStateObjectWriter;

    SerializedStateObject() { crc32::generate_table(_crc_table); }

    SerializedStateObject(const std::vector<uint8_t> &data, uint32_t checksum) : _data(data), _checksum(checksum) {
        crc32::generate_table(_crc_table);
    }

    // copy constructor
    SerializedStateObject(const SerializedStateObject &other) : SerializedStateObject(other._data, other._checksum) {}

    size_t size() const { return _data.size(); }
    bool isEmpty() const { return _data.empty(); }
    bool isValid() const { return _checksum == crc32::update(const_cast<SerializedStateObject *>(this)->_crc_table, 0, _data.data(), _data.size()); }

    uint32_t checksum() const { return _checksum; }
    const std::vector<uint8_t> &data() { return _data; }

    friend std::ostream &operator<<(std::ostream &os, const SerializedStateObject &state);

private:
    void updateChecksum() { _checksum = crc32::update(_crc_table, 0, _data.data(), _data.size()); }

private:
    std::vector<uint8_t> _data;
    uint32_t _crc_table[256];
    uint32_t _checksum = 0;
};

std::ostream &operator<<(std::ostream &os, const SerializedStateObject &state) {
    for (size_t i = 0; i < state._data.size(); i++) {
        os << state._data[i];
    }
    os << std::endl;
    return os;
};

class SerializedStateObjectReader {
    /*
     * Reads a SerializedStateObject instance from first the beginning of byte array.
     * Users can push data to a SerializedStateObject using SerializedStateObjectWriter, and
     * read information back using SerializedStateObjectReader just like in a file stream
     * read/write operation.
     *
     * Example:
     * SerializedStateObjectWriter writer(obj);
     * writer.push(myInt);
     * writer.push(myString);
     *
     *
     * SerializedStateObjectReader reader(obj);
     * int i = reader.read<int>();
     * std::string s = reader.read<std::string>();
     *
     * So the data is pushed and read in the same order.
     *
     * Note: this class assumes SerializedStateObject instance will be alive and available during the
     * whole lifecycle of the SerializedStateObjectReader. Destroying the SerializedStateObject instance
     * before the SerializedStateObjectReader will cause undefined behaviour!
     */
public:
    SerializedStateObjectReader(uint8_t *data, size_t totalSize) {
        data = data;
        totalSize = totalSize;
    }

    SerializedStateObjectReader(const SerializedStateObject &object) {
        data = const_cast<SerializedStateObject &>(object)._data.data();
        totalSize = const_cast<SerializedStateObject &>(object)._data.size();
    }

    // A readonly interface for a SerializedStateObject
    template <typename T, std::vector<T> (*deserializer)(const uint8_t *, size_t &)> std::vector<T> readUserVector() {
        /*
         * Reads a contigous vector composed by custom user type T.
         */

        // certify we can read
        checkRemainingBytes();

        size_t numRead = 0;
        auto result = deserializer(data, numRead);
        data += numRead;
        totalSize -= numRead;
        return result;
    }

    template <typename T> T read() {
        // read basic whole types or structs

        // certify we can read
        checkRemainingBytes();

        T value = *(T *)data;
        data += sizeof(T);
        totalSize -= sizeof(T);
        return value;
    }

    size_t remainingBytes() const { return totalSize; }
    bool isEmpty() const { return totalSize == 0; }

private:
    void checkRemainingBytes() {
        if (totalSize == 0) {
            throw new std::runtime_error("trying to read from an empty reader");
        }
    }

private:
    size_t totalSize = 0;
    uint8_t *data = nullptr;
};

class SerializedStateObjectWriter {
public:
    SerializedStateObjectWriter(SerializedStateObject &obj) : _object(obj) {
        /*
         * Writes data to SerializedStateObjectWriter instance to the end of its byte array.
         * Note: this class assumes SerializedStateObject instance will be alive and available during the
         * whole lifecycle of the SerializedStateObjectReader. Destroying the SerializedStateObject instance
         * before the SerializedStateObjectReader will cause undefined behaviour!
         */
    }

    virtual ~SerializedStateObjectWriter() { close(); }

    void close() {
        // update checksum
        object().updateChecksum();
    }

    template <typename T, void (*serializer)(const std::vector<T> &, std::vector<uint8_t> &)> void writeUserVector(const std::vector<T> &input) {
        /*
         * Writes an user vector to the serialized object. User vectors can have any memory layout and thereforee
         * need a custom serializer passed as template argument
         */
        serializer(input, object()._data);
    }

    template <typename T> void write(T type) {
        // push basic whole types or structs
        object()._data.insert(object()._data.end(), (uint8_t *)&type, (uint8_t *)&type + sizeof(T));
    }

private:
    SerializedStateObject &object() { return const_cast<SerializedStateObject &>(_object); }

private:
    const SerializedStateObject &_object;
};

template <> void SerializedStateObjectWriter::write(const std::string &type) {
    // push basic whole types or structs
    object()._data.insert(object()._data.end(), type.begin(), type.end());
}

}; // namespace vptree
