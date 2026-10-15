// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.support:math_check;

import dpl;

export namespace dpl::test {
inline namespace support {
constexpr bool finite_math_only() noexcept {
    if consteval {
        return false;
    } else {
        // Implementation defined, may not work everywhere
        float volatile nan = dpl::bit_cast<float>(-1);
        return nan == nan;
    }
}

constexpr bool ieee_denormal() noexcept {
    if consteval {
        return true;
    } else {
        // Implementation defined, may not work everywhere
        float volatile denormal = dpl::bit_cast<float>(0x00800000) * 0.5f;
        return denormal != 0.0f;
    }
}
} // namespace support
} // namespace dpl::test
