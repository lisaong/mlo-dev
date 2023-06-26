#include <cmath>
#include <limits>
#include <cstdint>
#include <algorithm>

std::uint64_t ULPDiff(float a, float b)
{
    // Make sure a is the larger number
    if (std::abs(b) > std::abs(a))
    {
        std::swap(a, b);
    }

    // Convert float to int for bitwise operations
    std::uint32_t ia = reinterpret_cast<std::uint32_t &>(a);
    std::uint32_t ib = reinterpret_cast<std::uint32_t &>(b);

    // Handle cases with different signs
    if ((ia < 0) != (ib < 0))
    {
        if (a == b)
            return 0; // a == b == 0
        else
            return std::numeric_limits<std::uint64_t>::max(); // ULP is max
    }

    // Calculate ULP difference
    std::uint64_t ulps = ia > ib ? ia - ib : ib - ia;
    return ulps;
}

// float toFloatPrecision(std::uint64_t ulp)
// {
//     // compute in number of significant digits
//     constexpr float logBase2of10 = 3.32;
//     constexpr std::uint64_t floatSignificantBits = 24; // 53 for double
//     return (floatSignificantBits - ulp) / logBase2of10;
// }