#include "tensor.hpp"

namespace MiniVLLM
{
    // overload the << operator to pring DataType directly
    std::ostream &operator<<(std::ostream &os, DataType type)
    {
        switch (type)
        {
        case DataType::FLOAT32:
            return os << "FLOAT32";
        case DataType::FLOAT16:
            return os << "FLOAT16";
        case DataType::BFLOAT16:
            return os << "BFLOAT16";
        case DataType::INT8:
            return os << "INT8";
        case DataType::INT16:
            return os << "INT16";
        case DataType::INT32:
            return os << "INT32";
        case DataType::INT64:
            return os << "INT64";
        case DataType::UINT8:
            return os << "UINT8";
        case DataType::UINT16:
            return os << "UINT16";
        case DataType::UINT32:
            return os << "UINT32";
        case DataType::UINT64:
            return os << "UINT64";
        }

        return os << "UNKNOWN";
    }

    DataType stringToDataType(const std::string &dtype)
    {
        if (dtype == "F32")
            return DataType::FLOAT32;
        if (dtype == "F16")
            return DataType::FLOAT16;
        if (dtype == "BF16")
            return DataType::BFLOAT16;
        if (dtype == "I8")
            return DataType::INT8;
        if (dtype == "I16")
            return DataType::INT16;
        if (dtype == "I32")
            return DataType::INT32;
        if (dtype == "I64")
            return DataType::INT64;
        if (dtype == "U8")
            return DataType::UINT8;
        if (dtype == "U16")
            return DataType::UINT16;
        if (dtype == "U32")
            return DataType::UINT32;
        if (dtype == "U64")
            return DataType::UINT64;

        throw std::runtime_error("Unsupported data type: " + dtype);
    }
}
