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
#include "kernels.hpp"

namespace
{
    constexpr float kRmsNormEpsilon = 1.0e-5f;

    void checkCuda(cudaError_t status, const char *operation)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(operation) + ": " + cudaGetErrorString(status));
        }
    }

    template <typename T>
    class DeviceBuffer
    {
    public:
        explicit DeviceBuffer(int elementCount) : elementCount_(elementCount)
        {
            checkCuda(cudaMalloc(reinterpret_cast<void **>(&data_), elementCount_ * sizeof(T)), "cudaMalloc");
        }

        ~DeviceBuffer()
        {
            cudaFree(data_);
        }

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
        std::vector<float> weights;
    };

    float rmsNormReference(float input, float weight, double sumOfSquares, int hiddenSize)
    {
        const double rms = std::sqrt(sumOfSquares / static_cast<double>(hiddenSize) + kRmsNormEpsilon);
        return static_cast<float>(static_cast<double>(input) / rms * static_cast<double>(weight));
    }

    bool almostEqual(float actual, float expected, float absTolerance, float relTolerance)
    {
        const float difference = std::fabs(actual - expected);
        return difference <= absTolerance + relTolerance * std::max(std::fabs(actual), std::fabs(expected));
    }

    template <typename T>
    std::vector<T> convertToStorage(const std::vector<float> &values);

    template <>
    std::vector<float> convertToStorage<float>(const std::vector<float> &values)
    {
        return values;
    }

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
    float toFloat(float value)
    {
        return value;
    }

    template <>
    float toFloat(__nv_bfloat16 value)
    {
        return __bfloat162float(value);
    }

    template <typename T>
    bool runTest(const TestCase &test, MiniVLLM::DType dtype, const char *dtypeName, float absTolerance, float relTolerance)
    {
        if (test.hiddenSize == 0 || test.input.size() != test.numTokens * test.hiddenSize || test.weights.size() != test.hiddenSize)
            throw std::invalid_argument("invalid RMSNorm test case: " + test.name);

        const int elementCount = test.numTokens * test.hiddenSize;
        const MiniVLLM::ModelConfig config = {
            .DTYPE = dtype,
            .HIDDEN_SIZE = test.hiddenSize,
            .RMS_NORM_EPS = kRmsNormEpsilon,
        };

        const std::vector<T> input = convertToStorage<T>(test.input);
        const std::vector<T> weights = convertToStorage<T>(test.weights);
        DeviceBuffer<T> dInput(elementCount);
        DeviceBuffer<T> dOutput(elementCount);
        DeviceBuffer<T> dWeights(test.hiddenSize);
        dInput.copyFromHost(input);
        dWeights.copyFromHost(weights);

        MiniVLLM::launchRMSNorm(config, test.numTokens, dOutput.get(), dInput.get(), dWeights.get());
        checkCuda(cudaGetLastError(), "launchRMSNorm");
        checkCuda(cudaDeviceSynchronize(), "RMSNorm execution");

        const std::vector<T> output = dOutput.copyToHost();
        for (int token = 0; token < test.numTokens; ++token)
        {
            double sumOfSquares = 0.0;
            for (int dimension = 0; dimension < test.hiddenSize; ++dimension)
            {
                const float value = toFloat(input[token * test.hiddenSize + dimension]);
                sumOfSquares += static_cast<double>(value) * value;
            }

            for (int dimension = 0; dimension < test.hiddenSize; ++dimension)
            {
                const int index = token * test.hiddenSize + dimension;
                const float expected = rmsNormReference(
                    toFloat(input[index]), toFloat(weights[dimension]), sumOfSquares, test.hiddenSize);
                const float actual = toFloat(output[index]);
                if (!almostEqual(actual, expected, absTolerance, relTolerance))
                {
                    std::cerr << "FAIL " << test.name << " [" << dtypeName << "] (token " << token << ", dimension " << dimension
                              << "): expected " << expected << ", got " << actual << '\n';
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
        std::uniform_real_distribution<float> inputDistribution(-10.0f, 10.0f);
        std::uniform_real_distribution<float> weightDistribution(0.25f, 1.75f);

        TestCase test{std::move(name), numTokens, hiddenSize, {}, {}};
        test.input.resize(numTokens * hiddenSize);
        test.weights.resize(hiddenSize);
        for (float &value : test.input)
            value = inputDistribution(generator);
        for (float &value : test.weights)
            value = weightDistribution(generator);
        return test;
    }
}

int main()
{
    try
    {
        std::vector<TestCase> tests = {
            {"single element", 1, 1, {3.0f}, {2.0f}},
            {"mixed signs and weights", 1, 10,
             {1.25f, -2.80f, 3.14f, -4.75f, 5.50f, -6.33f, 7.91f, -8.62f, 9.27f, -10.48f},
             {0.95f, 1.10f, 0.87f, 1.23f, 0.76f, 1.05f, 1.18f, 0.91f, 1.30f, 0.84f}},
            {"zero input", 3, 32, std::vector<float>(96, 0.0f), std::vector<float>(32, 1.0f)},
            makeRandomCase("multiple tokens", 5, 257, 42),
            makeRandomCase("block boundary: 1023", 2, 1023, 43),
            makeRandomCase("block boundary: 1024", 2, 1024, 44),
            makeRandomCase("block boundary: 1025", 2, 1025, 45),
        };

        bool passed = true;
        for (const TestCase &test : tests)
        {
            passed = runTest<float>(test, MiniVLLM::DType::Float32, "float32", 1.0e-5f, 1.0e-5f) && passed;
            passed = runTest<__nv_bfloat16>(test, MiniVLLM::DType::BFloat16, "bfloat16", 1.0e-2f, 1.0e-2f) && passed;
        }
        return passed ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "RMSNorm test error: " << error.what() << '\n';
        return 1;
    }
}
