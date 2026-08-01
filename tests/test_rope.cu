#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "dtype.hpp"
#include "kernels_api.hpp"

namespace
{
    constexpr float kTableTolerance = 1.0e-5f;

    void checkCuda(cudaError_t status, const char *operation)
    {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }

    template <typename T>
    class DeviceBuffer
    {
    public:
        explicit DeviceBuffer(int elementCount) : elementCount_(elementCount)
        {
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&data_), elementCount_ * sizeof(T)), "cudaMalloc");
        }

        ~DeviceBuffer() { cudaFree(data_); }

        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        T *get() const { return data_; }

        void copyFromHost(const std::vector<T> &source)
        {
            checkCuda(cudaMemcpy(data_, source.data(), elementCount_ * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy host to device");
        }

        std::vector<T> copyToHost() const
        {
            std::vector<T> result(elementCount_);
            checkCuda(cudaMemcpy(result.data(), data_, elementCount_ * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy device to host");
            return result;
        }

    private:
        int elementCount_;
        T *data_ = nullptr;
    };

    struct TestCase
    {
        std::string name;
        int numTokens;
        int headDim;
        float base;
        std::vector<float> input;
    };

    bool almostEqual(float actual, float expected, float absTolerance, float relTolerance)
    {
        const float difference = std::fabs(actual - expected);
        return difference <= absTolerance + relTolerance * std::max(std::fabs(actual), std::fabs(expected));
    }

    void createReferenceTables(const TestCase &test, std::vector<float> &cosTable, std::vector<float> &sinTable)
    {
        const int pairCount = test.headDim / 2;
        cosTable.resize(test.numTokens * pairCount);
        sinTable.resize(test.numTokens * pairCount);
        for (int position = 0; position < test.numTokens; ++position)
        {
            for (int pair = 0; pair < pairCount; ++pair)
            {
                const float exponent = 2.0f * static_cast<float>(pair) / static_cast<float>(test.headDim);
                const float theta = static_cast<float>(position) / std::pow(test.base, exponent);
                const int index = position * pairCount + pair;
                cosTable[index] = std::cos(theta);
                sinTable[index] = std::sin(theta);
            }
        }
    }

    template <typename T>
    std::vector<T> convertToStorage(const std::vector<float> &values);

    template <>
    std::vector<float> convertToStorage<float>(const std::vector<float> &values) { return values; }

    template <>
    std::vector<__nv_bfloat16> convertToStorage<__nv_bfloat16>(const std::vector<float> &values)
    {
        std::vector<__nv_bfloat16> converted(values.size());
        for (int index = 0; index < values.size(); ++index)
            converted[index] = __float2bfloat16(values[index]);
        return converted;
    }

    template <typename T>
    float toFloat(T value);

    template <>
    float toFloat(float value) { return value; }

    template <>
    float toFloat(__nv_bfloat16 value) { return __bfloat162float(value); }

    template <typename T>
    bool runTest(const TestCase &test, MiniVLLM::DType dtype, const char *dtypeName, float embeddingTolerance)
    {
        if (test.numTokens == 0 || test.headDim == 0 || test.headDim % 2 != 0 || test.input.size() != test.numTokens * test.headDim)
            throw std::invalid_argument("invalid RoPE test case: " + test.name);

        const int elementCount = test.numTokens * test.headDim;
        const int tableElementCount = test.numTokens * (test.headDim / 2);
        const MiniVLLM::ModelConfig config = {
            .DTYPE = dtype,
            .HEAD_DIM = test.headDim,
            .MAX_POSITION_EMBEDDINGS = test.numTokens,
            .ROPE_THETA = test.base,
        };

        std::vector<float> expectedCos;
        std::vector<float> expectedSin;
        createReferenceTables(test, expectedCos, expectedSin);

        DeviceBuffer<float> dCosTable(tableElementCount);
        DeviceBuffer<float> dSinTable(tableElementCount);
        DeviceBuffer<T> dEmbeddings(elementCount);
        const std::vector<T> input = convertToStorage<T>(test.input);
        dEmbeddings.copyFromHost(input);

        MiniVLLM::launchCreateRoPETables(config, dCosTable.get(), dSinTable.get());
        checkCuda(cudaGetLastError(), "launchCreateRoPETables");
        checkCuda(cudaDeviceSynchronize(), "RoPE table creation");

        const std::vector<float> actualCos = dCosTable.copyToHost();
        const std::vector<float> actualSin = dSinTable.copyToHost();
        for (int index = 0; index < tableElementCount; ++index)
        {
            if (!almostEqual(actualCos[index], expectedCos[index], kTableTolerance, kTableTolerance) ||
                !almostEqual(actualSin[index], expectedSin[index], kTableTolerance, kTableTolerance))
            {
                std::cerr << "FAIL " << test.name << " [" << dtypeName << "] table entry " << index << '\n';
                return false;
            }
        }

        MiniVLLM::launchRoPEEmbeddings(config, test.numTokens, dEmbeddings.get(), dCosTable.get(), dSinTable.get());
        checkCuda(cudaGetLastError(), "launchRoPEEmbeddings");
        checkCuda(cudaDeviceSynchronize(), "RoPE embedding execution");

        const std::vector<T> actual = dEmbeddings.copyToHost();
        for (int position = 0; position < test.numTokens; ++position)
        {
            for (int pair = 0; pair < test.headDim / 2; ++pair)
            {
                const int tableIndex = position * (test.headDim / 2) + pair;
                const int embeddingIndex = tableIndex * 2;
                const float first = toFloat(input[embeddingIndex]);
                const float second = toFloat(input[embeddingIndex + 1]);
                const float expectedFirst = first * expectedCos[tableIndex] - second * expectedSin[tableIndex];
                const float expectedSecond = first * expectedSin[tableIndex] + second * expectedCos[tableIndex];

                if (!almostEqual(toFloat(actual[embeddingIndex]), expectedFirst, embeddingTolerance, embeddingTolerance) ||
                    !almostEqual(toFloat(actual[embeddingIndex + 1]), expectedSecond, embeddingTolerance, embeddingTolerance))
                {
                    std::cerr << "FAIL " << test.name << " [" << dtypeName << "] (position " << position << ", pair " << pair << ")\n";
                    return false;
                }
            }
        }

        std::cout << "PASS " << test.name << " [" << dtypeName << "]\n";
        return true;
    }

    TestCase makeRandomCase(std::string name, int numTokens, int headDim, float base, unsigned int seed)
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
        TestCase test{std::move(name), numTokens, headDim, base, std::vector<float>(numTokens * headDim)};
        for (float &value : test.input)
            value = distribution(generator);
        return test;
    }
}

int main()
{
    try
    {
        std::vector<TestCase> tests = {
            {"identity at position zero", 1, 4, 10000.0f, {1.0f, 2.0f, 3.0f, 4.0f}},
            {"multiple positions", 3, 4, 10000.0f, {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, -1.0f, -3.0f, 4.0f, 5.0f, 6.0f, -7.0f, 8.0f}},
            {"zero input", 4, 32, 10000.0f, std::vector<float>(128, 0.0f)},
            makeRandomCase("Llama 3 base", 16, 64, 500000.0f, 1337),
            makeRandomCase("block boundary: 2046", 2, 2046, 10000.0f, 1338),
            makeRandomCase("block boundary: 2048", 2, 2048, 10000.0f, 1339),
            makeRandomCase("block boundary: 2050", 2, 2050, 10000.0f, 1340),
        };

        bool passed = true;
        for (const TestCase &test : tests)
        {
            passed = runTest<float>(test, MiniVLLM::DType::Float32, "float32", 1.0e-5f) && passed;
            passed = runTest<__nv_bfloat16>(test, MiniVLLM::DType::BFloat16, "bfloat16", 1.5e-2f) && passed;
        }
        return passed ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "RoPE test error: " << error.what() << '\n';
        return 1;
    }
}
