#ifndef __VPTREE_SERIALIZABLE_HPP__
#define __VPTREE_SERIALIZABLE_HPP__

#include <cstring>
#include <ostream>
#include <string>
#include <vector>

namespace vptree {

struct SerializedState {
    std::vector<uint8_t> data;
    uint8_t checksum = 0;

    SerializedState() = default;
    SerializedState(const std::vector<uint8_t> &data, uint8_t checksum) : data(data), checksum(checksum) {}
    // copy constructor
    SerializedState(const SerializedState &other) : data(other.data), checksum(other.checksum) {}

    void reserve(size_t size) { data.reserve(size); }

    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }

    template <typename T> void push(const T &value) { data.insert(data.end(), (uint8_t *)&value, (uint8_t *)&value + sizeof(T)); }

    void push_by_size(const void *origin, size_t size) {
        size_t insert_pos = data.size();
        data.resize(data.size() + size);
        uint8_t *dest = &data[insert_pos];
        std::memcpy(dest, origin, size);
    }

    template <typename T> T pop() {
        T value = *(T *)&data[data.size() - sizeof(T)];
        data.resize(data.size() - sizeof(T));
        return value;
    }

    void pop_by_size(void *dest, size_t size) {
        uint8_t *origin = &data[data.size() - size];
        std::memcpy(dest, origin, size);
        data.resize(data.size() - size);
    }

    SerializedState operator+(const SerializedState &other) const {
        SerializedState result;
        result.data.reserve(data.size() + other.data.size());
        result.data.insert(result.data.end(), data.begin(), data.end());
        result.data.insert(result.data.end(), other.data.begin(), other.data.end());
        result.buildChecksum();
        return result;
    }

    SerializedState &operator+=(const SerializedState &other) {
        data.reserve(data.size() + other.data.size());
        data.insert(data.end(), other.data.begin(), other.data.end());
        buildChecksum();
        return *this;
    }

    bool isValid() const { return checksum == calculate_check_sum(data); }

    void buildChecksum() { checksum = calculate_check_sum(data); }

    friend std::ostream &operator<<(std::ostream &os, const SerializedState &state);

    private:
    uint8_t calculate_check_sum(const std::vector<uint8_t> &data) const {
        // create an uint8 bitmaks in which bits are alternating 0 and 1
        uint32_t sum = 0;
        for (uint8_t byte : data) {
            sum += byte;
        }
        return sum % 256;
    }
};

class ISerializable {
    public:
    virtual SerializedState serialize() const = 0;
    virtual void deserialize(const SerializedState &state) = 0;
};

std::ostream &operator<<(std::ostream &os, const SerializedState &state) {
    for (int i = 0; i < state.data.size(); i++) {
        os << state.data[i];
    }
    os << std::endl;
}

}; // namespace vptree

#endif
