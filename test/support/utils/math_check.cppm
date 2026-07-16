// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test:utils.math_check;

import dpl;

export namespace dpl::test {

constexpr bool ieee_denormal() noexcept {
    if consteval {
        return true;
    } else {
        float volatile denorm =
            dpl::bit_cast<float>(dpl::floating_point_traits<float>::signbit >>
                (dpl::floating_point_traits<float>::width - 1));
        return (denorm + 0.0f) == denorm;
    }
}

constexpr bool finite_math_only() noexcept {
    if consteval {
        return false;
    } else {
        float volatile inf = dpl::datapar::infinity_v<float>;
        return inf + 1.0f != inf;
    }
}

} // namespace dpl::test
