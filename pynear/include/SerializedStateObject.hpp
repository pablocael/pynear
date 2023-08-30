#pragma onc

#include "crc32.hpp"
#include <cstring>
#include <ostream>
#include <string>
#include <vector>

namespace vptree {

uint8_t calculate_check_sum(const std::vector<uint8_t> &data) const {
    // create an uint8 bitmaks in which bits are alternating 0 and 1
    uint32_t sum = 0;
    for (uint8_t byte : data) {
        sum += byte;
    }
    return sum % 256;
};

class SerializedStateObject {
    /*
     * Stores a serialized object in a byte array. The object can be read and written using
     * SerializedStateObjectReader and SerializedStateObjectWriter.
     */
public:
    friend class SerializedStateObjectReader;
    friend class SerializedStateObjectWriter;

    SerializedStateObject() { crc32::generate_table(_crc_table); }

    SerializedStateObject(const std::vector<uint8_t> &data, uint8_t checksum) : _data(data), _checksum(checksum) {
        crc32::generate_table(_crc_table);
    }

    // copy constructor
    SerializedStateObject(const SerializedStateObject &other) : SerializedStateObject(other._data, other._checksum) {}

    size_t size() const { return _data.size(); }
    bool isEmpty() const { return _data.empty(); }
    bool isValid() const { return _checksum == calculate_check_sum(_data); }
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
     * whole lifecycle of the SerializedStateObjectReader.
     */
public:
    SerializedStateObjectReader(uint8_t *data) { data = data; }
    SerializedStateObjectReader(const SerializedStateObject &object) { data = const_cast<SerializedStateObject &>(object)._data.data(); }

    // A readonly interface for a SerializedStateObject
    template <typename T, std::vector<T> (*deserializer)(const uint8_t *, size_t &)> std::vector<T> readUserVector() {
        /*
         * Readss a contigous vector composed by custom user type T.
         */
        size_t numRead = 0;
        auto result = deserializer(data, numRead);
        data += numRead;
        return result;
    }

    template <typename T> T read() {
        // read basic whole types or structs
        T value = *(T *)data;
        data += sizeof(T);
        return value;
    }

private:
    uint8_t *data = nullptr;
};

class SerializedStateObjectWriter {
public:
    SerializedStateObjectWriter(SerializedStateObject &obj) : _object(obj) {
        /*
         * Warning: this class assumes SerializedStateObject instance will be alive and available during the whole
         * lifecycle of the SerializedStateObjectWriter.
         */
    }

    template <typename T, void (*serializer)(const std::vector<T> &, std::vector<uint8_t> &)> void pushUserVector(const std::vector<T> &input) {
        /*
         * Writes an user vector to the serialized object. User vectors can have any memory layout and thereforee
         * need a custom serializer passed as template argument
         */
        serializer(input, object()._data);
    }

    template <typename T, std::vector<T> (*deserializer)(const uint8_t *, size_t &)> std::vector<T> popUserVector() {
        // pop for array of custom complex types with custom deserializer

        size_t numRead = 0;

        std::vector<uint8_t> &data = object()._data;
        auto result = deserializer(data.data(), numRead);
        data.resize(data.size() - numRead);
        return result;
    }

    template <typename T> void push(T type) {
        // push basic whole types or structs
        object()._data.insert(object()._data.end(), (uint8_t *)&type, (uint8_t *)&type + sizeof(T));
    }

    template <typename T> T pop() {
        // pop basic whole types or structs
        T value = *(T *)object()._data.data();
        object()._data.erase(object()._data.begin(), object()._data.begin() + sizeof(T));
        return value;
    }

private:
    SerializedStateObject &object() { return const_cast<SerializedStateObject &>(_object); }

private:
    const SerializedStateObject &_object;
};

}; // namespace vptree
