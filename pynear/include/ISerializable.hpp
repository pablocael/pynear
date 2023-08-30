#pragma once

#include <cstring>
#include <ostream>
#include <string>
#include <vector>

#include "SerializedStateObject.hpp"

namespace vptree {

class ISerializable {
public:
    virtual SerializedStateObject serialize() const = 0;
    virtual void deserialize(const SerializedStateObject &state) = 0;
};

}; // namespace vptree
