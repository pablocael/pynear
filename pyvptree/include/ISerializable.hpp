#ifndef __VPTREE_SERIALIZABLE_HPP__
#define __VPTREE_SERIALIZABLE_HPP__

#include <string>
#include <vector>

namespace vptree {

struct SerializedState {
    std::vector<uint8_t> data;
    uint8_t checksum = 0;

    SerializedState() = default;
    SerializedState(const std::vector<uint8_t> &data, uint8_t checksum) : data(data), checksum(checksum) {}

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

    private:
    uint8_t calculate_check_sum(const std::vector<uint8_t> &data) const {
        // create an uint8 bitmaks in which bits are alternating 0 and 1
        uint32_t sum = 0;
        for (uint8_t byte : data) {
            sum += byte;
        }
        return sum % 256;
    }
    void buildChecksum() { checksum = calculate_check_sum(data); }
};

class ISerializable {
    public:
    virtual SerializedState serialize() const = 0;
    virtual void deserialize(const SerializedState &state) = 0;
};
}; // namespace vptree

#endif
