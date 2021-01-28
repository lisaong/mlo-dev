## Hello GEMM

By the end of this tutorial, you will learn how to:
* Implement a Naive GEMM (Generalized Matrix Multiply) function using RoboCode's Domain Specific Language (DSL)
* Call the function from a C++ executable
* Verify the results of an example matrix-matrix multiplication

### Pre-requisites

* This tutorial assumes you already have RoboCode installed. If not, you can find the instructions under the [Installing RoboCode](installing.html) section. 
* You should also be familiar with writing C++.

### A naive GEMM algorithm

Let's consider the example of multiplying matrices A and B. 

```
C = A * B
```

A Naive algorithm typically contains 3 nested for loops:
```
 for each value of i
   for each value of j
     take the sum over k of A(i,k) * B(k,j) 
```

In C++, this logic can be expressed as:

```cpp
// C = A * B
void MM(const Matrix& A, const Matrix& B, Matrix& C)
{
    for(int i = 0; i < C.NumRows(); i++)
        for(int j = 0; j < C.NumColumns(); j++)
            for(int k = 0; k < A.NumColumns(); k++)
                C(i, j) += A(i, k) * B(k, j);
}
```

#### RoboCode DSL

Expressed in the RoboCode DSL, the above algorithm can be written as:

```cpp
// Express the algorithm using the RoboCode DSL for a given domain configuration
void RoboCodeSampleFunction(const Domain& domain, Matrix A, Matrix B, Matrix C)
{
    // Define the loop nest
    Nest nest(domain);

    // Get the loop nest indices
    auto indices = nest.GetIndices();
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // Set the loop nest kernel
    nest.Set([&]() { C(i, j) += A(i, k) * B(k, j); });
}
```

We are using the `Nest` object, with a kernel that performs the add-multiply computation of the innermost loop. The `domain` specifies a `M, N, K` block size configuration for the `i`, `j`, and `k` indices. 

RoboCode will use this information to emit (generate) a function that performs the nest computation. So, `900, 1600, 100` means `i` ranges from 0-899, `j` ranges from 0-1599, and `k` from 0-99.

1. Create file called `domains.csv` with the following contents. You can also download it from here (link TBD).

```
i, j, k
900,1600,100
```

2. Create a file called `hello_gemm_generator.cpp` with the code below. You can also download it from here (link TBD).

```cpp
#include <RoboCode.h>
#include <RoboCodeDomain.h>
#include <RoboCodeEmitterHelper.h>

#include <value/include/Nest.h>

#include <string>
#include <tuple>

using namespace robocode;
using namespace robocode::experimental;
using namespace robocode::parameter;
using namespace robocode::utilities;

// Domain configurations filled in ParseRoboCodeCommandLineOptions (defined in domains.csv)
static DomainListParameter Domains;

// Express the algorithm using the RoboCode DSL for a given domain configuration
void RoboCodeSampleFunction(const Domain& domain, Matrix A, Matrix B, Matrix C)
{
    // Define the loop nest
    Nest nest(domain);

    // Get the loop nest indices
    auto indices = nest.GetIndices();
    auto i = indices[0];
    auto j = indices[1];
    auto k = indices[2];

    // Set the loop nest kernel
    nest.Set([&]() { C(i, j) += A(i, k) * B(k, j); });
}

// Define the name for the function to be emitted
std::string GetSampleFunctionName(const Domain& domain)
{
    std::string functionName = "helloWorldGEMM";
    for (const auto& domainDim : domain)
    {
        functionName += "_" + std::to_string(domainDim.Size());
    }
    return functionName;
}

// Define the argument memory layout for the function to be emitted
auto GetSampleArgLayouts(const Domain& domain)
{
    auto M = domain[0].Size();
    auto N = domain[1].Size();
    auto K = domain[2].Size();

    MemoryLayout layoutA{ MemoryShape{ M, K } };
    MemoryLayout layoutB{ MemoryShape{ K, N } };
    MemoryLayout layoutC{ MemoryShape{ M, N } };
    return std::tuple{ layoutA, layoutB, layoutC };
}

int main(int argc, const char** argv)
{
    ParseRoboCodeCommandLineOptions(argc, argv);
    EmitRoboCodeFunction<float>(Domains,
                                RoboCodeSampleFunction,
                                GetSampleFunctionName,
                                GetSampleArgLayouts);
}
```

#### Runner code

After defining the DSL, we will now walk through how to call the functions emitted for the above DSL code.

Create a file called `hello_gemm_runner.cpp` with the code below. You can also download it from here (link TBD).

```cpp

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
```

The code above demonstrates how to use the generated functions (specifically, for the `900, 1600, 100` configuration).

It does the following:
1. Loads data into the `A` and `B` matrices and zero-initializes the `C` matrix
2. Initializes the module
3. Calls the generated function `helloWorldGEMM_900_1600_100`, providing information about the memory layout
4. Computes timing information
5. De-initializes the module

#### Calling the RCC toolchain

By now you should have the following files in the same folder:

```
-a----         28/1/2021   6:12 pm             61 domains.csv
-a----         26/1/2021   3:41 pm           2523 hello_gemm_generator.cpp
-a----         28/1/2021   6:32 pm           4012 hello_gemm_runner.cpp
```

Run the `rcc.py` script. This will:
1. Emit functions implementing the DSL declared in `hello_gemm_generator.cpp` for each of the configurations specified in `domains.csv`
2. Compile the functions into a statically linked library, and
3. Build an executable with the runner code in `hello_gemm_runner.cpp`

Replace `path_to_robocode` with the path to your RoboCode git repository.

Windows:

From a "Developer Command Prompt", run powershell (so that the back-ticks work):

```
powershell
```

Run the script:
```
python path_to_robocode\build\install\bin\rcc.py `
    hello_gemm_generator.cpp `
    --domain domains.csv `
    --library_name hello_gemm `
    --output .\hello_gemm\ `
    --main hello_gemm_runner.cpp
```

MacOS/Linux:

```
python path_to_robocode/build/install/bin/rcc.py \
    hello_gemm_generator.cpp \
    --domain domains.csv \
    --library_name hello_gemm \
    --output ./hello_gemm/ \
    --main hello_gemm_runner.cpp
```

When the script completes, an executable called `hello_gemm_main` or `hello_gemm_main.exe` will be built in the `hello_gemm/main/build/Release/` folder.

#### Preparing the input data

Instead of using randomly generated data, let's try a simple matrix multiplication scenario. You can find a notebook hosted on Google Colaboratory [here](https://github.com/lisaong/data/blob/master/demos/Ninja_Cat_GEMM.ipynb).

Follow the instructions of the notebook to generate these files:
* A_matrix.csv
* B_matrix.csv

When prompted, download the files to the same location as the `*.cpp` files:

```
d-----         28/1/2021   6:57 pm                hello_gemm
-a----         28/1/2021   6:53 pm        2295358 A_matrix.csv  <---- (new)
-a----         28/1/2021   6:55 pm        4080538 B_matrix.csv  <---- (new)
-a----         28/1/2021   6:12 pm             61 domains.csv
-a----         26/1/2021   3:41 pm           2523 hello_gemm_generator.cpp
-a----         28/1/2021   6:32 pm           4012 hello_gemm_runner.cpp
```

#### Running the code

We are now ready to test the emitted GEMM function by calling running the `hello_gemm_main` executable on our input data:

Windows:

```
hello_gemm\main\build\Release\hello_gemm_main.exe
```

MacOS/Linux:

```
hello_gemm/main/build/Release/hello_gemm_main
```

When done, you should see some timing information for the emitted GEMM function:

```
Calling GEMM M=900, K=100, N=1600
Elapsed time: 198 msecs
```

Actual timings will depend on the system that you are running on.

There should be a file created called `C_matrix_robocode.csv` in that same location:

```
d-----         28/1/2021   6:57 pm                hello_gemm
-a----         28/1/2021   6:53 pm        2295358 A_matrix.csv
-a----         28/1/2021   6:55 pm        4080538 B_matrix.csv
-a----         28/1/2021   7:23 pm       10987145 C_matrix_robocode.csv  <---- (new)
-a----         28/1/2021   6:12 pm             61 domains.csv
-a----         26/1/2021   3:41 pm           2523 hello_gemm_generator.cpp
-a----         28/1/2021   6:32 pm           4012 hello_gemm_runner.cpp
```

#### Verifying the result

Finally, you can verify the contents of `C_matrix_robocode.csv` by uploading it to the Colaboratory environment, and running the remaining parts of that notebook. The notebook will load the C matrix into numpy, plot it as an image, and compare it with the results of [numpy.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html).

In the next section, we'll see how to add a simple optimization to improve the performance of our naive GEMM algorithm.