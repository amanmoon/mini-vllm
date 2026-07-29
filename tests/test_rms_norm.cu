#include <cuda_runtime.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <kernels.cuh>

constexpr float EPS = 1e-5f;

struct TestCase
{
    std::string name;
    std::vector<float> input;
    std::vector<float> normWeights;
};

void runTest(const TestCase &test)
{
    const int vectorDim = test.input.size();

    float *dInput = nullptr;
    float *dOutput = nullptr;
    float *dNormWeights = nullptr;

    cudaMalloc(&dInput, vectorDim * sizeof(float));
    cudaMalloc(&dOutput, vectorDim * sizeof(float));
    cudaMalloc(&dNormWeights, vectorDim * sizeof(float));

    cudaMemcpy(
        dInput,
        test.input.data(),
        vectorDim * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        dNormWeights,
        test.normWeights.data(),
        vectorDim * sizeof(float),
        cudaMemcpyHostToDevice);

    MiniVLLM::rootMeanSquareNorm<float>
        <<<1, 1024>>>(vectorDim,
                      dOutput,
                      dInput,
                      dNormWeights);

    cudaDeviceSynchronize();

    std::vector<float> gpuOutput(vectorDim);

    cudaMemcpy(
        gpuOutput.data(),
        dOutput,
        vectorDim * sizeof(float),
        cudaMemcpyDeviceToHost);

    // CPU reference
    float sum = 0.0f;
    for (float x : test.input)
        sum += x * x;

    float rms = std::sqrt(sum / vectorDim + EPS);

    bool passed = true;
    int failedIndex = -1;
    float expectedValue = 0.0f;

    for (int i = 0; i < vectorDim; i++)
    {
        float expected =
            (test.input[i] / rms) * test.normWeights[i];

        if (std::fabs(expected - gpuOutput[i]) > EPS)
        {
            passed = false;
            failedIndex = i;
            expectedValue = expected;
            break;
        }
    }

    std::cout << std::left << std::setw(30)
              << test.name;

    if (passed)
    {
        std::cout << "PASS\n";
    }
    else
    {
        std::cout << "FAIL\n";

        std::cout << "  Failed at index : "
                  << failedIndex << '\n';

        std::cout << "  Expected        : "
                  << expectedValue << '\n';

        std::cout << "  GPU             : "
                  << gpuOutput[failedIndex] << '\n';

        std::cout << "  Difference      : "
                  << std::fabs(expectedValue - gpuOutput[failedIndex])
                  << "\n\n";
    }

    cudaFree(dInput);
    cudaFree(dOutput);
    cudaFree(dNormWeights);
}

int main()
{
    std::vector<TestCase> tests =
    {
        {
            "Sequential",
            {1,2,3,4,5,6,7,8,9,10},
            {1,1,1,1,1,1,1,1,1,1}
        },

        {
            "Mixed Signs",
            {
                1.25f,-2.80f,3.14f,-4.75f,5.50f,
                -6.33f,7.91f,-8.62f,9.27f,-10.48f
            },
            {
                0.95f,1.10f,0.87f,1.23f,0.76f,
                1.05f,1.18f,0.91f,1.30f,0.84f
            }
        },

        {
            "All Ones",
            {1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1}
        },

        {
            "Negative Inputs",
            {-1,-2,-3,-4,-5,-6,-7,-8,-9,-10},
            {1,1,1,1,1,1,1,1,1,1}
        },

        {
            "Random Weights",
            {2,4,6,8,10,12,14,16,18,20},
            {0.3f,0.7f,1.1f,1.5f,0.2f,
             0.9f,1.8f,0.6f,1.3f,2.0f}
        },

        {
            "Tiny Values",
            {
                0.001f,-0.002f,0.003f,-0.004f,0.005f,
                -0.006f,0.007f,-0.008f,0.009f,-0.010f
            },
            {1,1,1,1,1,1,1,1,1,1}
        }
    };

    // Large random test (5000 dimensions)
    {
        constexpr int N = 5000;

        TestCase randomTest;
        randomTest.name = "Large Random (5000)";

        randomTest.input.resize(N);
        randomTest.normWeights.resize(N);

        std::mt19937 rng(42);

        std::uniform_real_distribution<float> inputDist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> weightDist(0.5f, 1.5f);

        for (int i = 0; i < N; i++)
        {
            randomTest.input[i] = inputDist(rng);
            randomTest.normWeights[i] = weightDist(rng);
        }

        tests.push_back(std::move(randomTest));
    }


    int passed = 0;

    for (const auto &test : tests)
    {
        runTest(test);
    }

    return 0;
}