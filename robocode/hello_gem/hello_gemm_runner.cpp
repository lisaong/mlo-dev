////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  RoboCode
//  File:     hello_gemm_runner.cpp
//  Authors:  Lisa Ong
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// Include the header file that declares our emitted functions
#include <hello_gemm.h>

#pragma region helpers

// Helper class to print timing information
class AutoTimer
{
public:
    AutoTimer() : _start(std::chrono::high_resolution_clock::now())
    {}
    ~AutoTimer()
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - _start).count();
        std::cout << "Elapsed time: " << duration << " msecs" << std::endl;
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
};

// Helper function to load data from a CSV file
// A CSV file can be generated from a numpy array in Python like this:
//    numpy.savetxt("file.csv", ndarray, delimiter=',')
template <auto NumElements>
bool LoadDataFromCSV(const std::string& csvPath, std::array<float, NumElements>& output)
{
    std::ifstream file(csvPath);
    if (!file.is_open())
    {
        std::cout << "Could not open file: " << csvPath << std::endl;
        return false;
    }

    auto iterator = output.begin();
    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            // Minimal error checking only
            try
            {
                *(iterator++) = std::stof(token);
            }
            catch (...)
            {
                std::cout << "Could not add token " << token << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Helper function to save data to a CSV file
// The CSV file can be loaded into a numpy array in Python like this:
//    ndarray = numpy.loadtxt("file.csv", delimiter=',')
template <auto NumElements>
void SaveDataToCSV(const std::array<float, NumElements>& input, int64_t rowStride, const std::string& csvPath)
{
    std::ofstream file(csvPath);
    for (auto i=0; i<input.size(); ++i)
    {
        file << input[i];

        // 1 row per line
        if ((i+1) % rowStride != 0)
        {
            file << ",";
        }
        else
        {
            file << std::endl;
        }
    }
    file.close();
}

#pragma endregion helpers

int main(int argc, const char** argv)
{
    helloWorldGEMM_900_1600_100_module_initialize();

    // Prepare our matrices
    constexpr int64_t M = 900;
    constexpr int64_t N = 1600;
    constexpr int64_t K = 100;
    const int64_t offset = 0;
    const int64_t columnStride = 1;

    std::array<float, M*K> A{};
    std::array<float, K*N> B{};
    std::array<float, M*N> C{};

    if (LoadDataFromCSV("A_matrix.csv", A) &&
        LoadDataFromCSV("B_matrix.csv", B))
    {
        std::cout << "Calling GEMM M=900, K=100, N=1600" << std::endl;
        {
            AutoTimer timer;

            helloWorldGEMM_900_1600_100(
                A.data() /* allocated */, A.data() /* aligned */, offset,
                M /* rows */, K /* cols */, K /* row stride */, columnStride,
                B.data() /* allocated */, B.data() /* aligned */, offset,
                K /* rows */, N /* cols */, N /* row stride */, columnStride,
                C.data() /* allocated */, C.data() /* aligned */, offset,
                M /* rows */, N /* cols */, N /* row stride */, columnStride);
        }

        SaveDataToCSV(C, N, "C_matrix_robocode.csv");
    }

    helloWorldGEMM_900_1600_100_module_deinitialize();
}
