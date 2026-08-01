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
        int hiddenSize;
        std::vector<float> input;
        std::vector<float> output;
    };

    bool almostEqual(float actual, float expected, float absTolerance, float relTolerance)
    {
        const float difference = std::fabs(actual - expected);
        return difference <= absTolerance + relTolerance * std::max(std::fabs(actual), std::fabs(expected));
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
    bool runTest(const TestCase &test, MiniVLLM::DType dtype, const char *dtypeName, float tolerance)
    {
        const int elementCount = test.numTokens * test.hiddenSize;
        if (test.numTokens == 0 || test.hiddenSize == 0 || test.input.size() != elementCount || test.output.size() != elementCount)
            throw std::invalid_argument("invalid residual-add test case: " + test.name);

        const MiniVLLM::ModelConfig config = {
            .DTYPE = dtype,
            .HIDDEN_SIZE = test.hiddenSize,
        };
        const std::vector<T> input = convertToStorage<T>(test.input);
        const std::vector<T> initialOutput = convertToStorage<T>(test.output);
        DeviceBuffer<T> dInput(elementCount);
        DeviceBuffer<T> dOutput(elementCount);
        dInput.copyFromHost(input);
        dOutput.copyFromHost(initialOutput);

        MiniVLLM::launchResidualAdd(config, test.numTokens, dInput.get(), dOutput.get());
        checkCuda(cudaGetLastError(), "launchResidualAdd");
        checkCuda(cudaDeviceSynchronize(), "residual add execution");

        const std::vector<T> output = dOutput.copyToHost();
        for (int token = 0; token < test.numTokens; ++token)
        {
            for (int dimension = 0; dimension < test.hiddenSize; ++dimension)
            {
                const int index = token * test.hiddenSize + dimension;
                const float expected = toFloat(initialOutput[index]) + toFloat(input[index]);
                const float actual = toFloat(output[index]);
                if (!almostEqual(actual, expected, tolerance, tolerance))
                {
                    std::cerr << "FAIL " << test.name << " [" << dtypeName << "] (token " << token
                              << ", dimension " << dimension << "): expected " << expected << ", got " << actual << '\n';
                    return false;
                }
            }
        }

        std::cout << "PASS " << test.name << " [" << dtypeName << "]\n";
        return true;
    }

    TestCase makeRandomCase(std::string name, int numTokens, int hiddenSize, unsigned int seed)
    {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
        const int elementCount = numTokens * hiddenSize;
        TestCase test{std::move(name), numTokens, hiddenSize, std::vector<float>(elementCount), std::vector<float>(elementCount)};
        for (float &value : test.input)
            value = distribution(generator);
        for (float &value : test.output)
            value = distribution(generator);
        return test;
    }
}

int main()
{
    try
    {
        std::vector<TestCase> tests = {
            {"single element", 1, 1, {2.5f}, {-1.0f}},
            {"multiple tokens and mixed signs", 3, 4,
             {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f},
             {-0.5f, 1.5f, -2.5f, 3.5f, -4.5f, 5.5f, -6.5f, 7.5f, -8.5f, 9.5f, -10.5f, 11.5f}},
            {"zero input", 3, 32, std::vector<float>(96, 0.0f), std::vector<float>(96, 2.0f)},
            makeRandomCase("multiple tokens", 5, 257, 53),
            makeRandomCase("block boundary: 1023", 2, 1023, 54),
            makeRandomCase("block boundary: 1024", 2, 1024, 55),
            makeRandomCase("block boundary: 1025", 2, 1025, 56),
        };

        bool passed = true;
        for (const TestCase &test : tests)
        {
            passed = runTest<float>(test, MiniVLLM::DType::Float32, "float32", 1.0e-5f) && passed;
            passed = runTest<__nv_bfloat16>(test, MiniVLLM::DType::BFloat16, "bfloat16", 1.0e-2f) && passed;
        }
        return passed ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Residual-add test error: " << error.what() << '\n';
        return 1;
    }
}
