#include "dtype.hpp"

namespace MiniVLLM
{
    // overload the << operator to pring DataType directly
    std::ostream &operator<<(std::ostream &os, DType type)
    {
        switch (type)
        {
        case DType::Float32:
            return os << "Float32";
        case DType::Float16:
            return os << "Float16";
        case DType::BFloat16:
            return os << "BFloat16";
        case DType::Int8:
            return os << "Int8";
        case DType::Int16:
            return os << "Int16";
        case DType::Int32:
            return os << "Int32";
        case DType::Int64:
            return os << "Int64";
        case DType::UInt8:
            return os << "UInt8";
        case DType::UInt16:
            return os << "UInt16";
        case DType::UInt32:
            return os << "UInt32";
        case DType::UInt64:
            return os << "UInt64";
        }

        return os << "Unknown";
    }

    DType stringToDType(const std::string &dtype)
    {
        if (dtype == "F32")
            return DType::Float32;
        if (dtype == "F16")
            return DType::Float16;
        if (dtype == "BF16")
            return DType::BFloat16;
        if (dtype == "I8")
            return DType::Int8;
        if (dtype == "I16")
            return DType::Int16;
        if (dtype == "I32")
            return DType::Int32;
        if (dtype == "I64")
            return DType::Int64;
        if (dtype == "U8")
            return DType::UInt8;
        if (dtype == "U16")
            return DType::UInt16;
        if (dtype == "U32")
            return DType::UInt32;
        if (dtype == "U64")
            return DType::UInt64;

        throw std::runtime_error("Unsupported data type: " + dtype);
    }
}
