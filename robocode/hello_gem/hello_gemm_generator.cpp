////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  RoboCode
//  File:     hello_gemm_generator.cpp
//  Authors:  Mason Remy, Lisa Ong
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <RoboCode.h>
#include <RoboCodeDomain.h>
#include <RoboCodeEmitterHelper.h>

#include <value/include/Nest.h>

#include <cassert>
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
    assert(domain.size() == 3 && "Domain must have 3 dimensions for GEMM");

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
    assert(domain.size() == 3 && "Domain must have 3 dimensions for GEMM");

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
    assert(domain.size() == 3 && "Domain must have 3 dimensions for GEMM");

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