// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.support:math_check;

import dpl;

export namespace dpl::test {
inline namespace support {
template <dpl::floating_point_like T = float>
constexpr bool finite_math_only() noexcept {
    if consteval {
        return false;
    } else {
        // Implementation defined, may not work everywhere
        using bitset_t = dpl::bit_representation_t<T>;
        T volatile nan = dpl::bit_cast<T>(~bitset_t());
        return nan == nan;
    }
}

template <dpl::floating_point_like T = float>
constexpr bool flush_to_zero() noexcept {
    if consteval {
        return true;
    } else {
        // Implementation defined, may not work everywhere
        auto const volatile bits =
            dpl::to_underlying(dpl::floating_point_traits<T>::mantissa_mask);
        using bits_t = dpl::decay_t<decltype(bits)>;
        auto const tiny = dpl::bit_cast<T>(static_cast<bits_t>(bits));
        return tiny * static_cast<T>(1) == static_cast<T>(0);
    }
}
} // namespace support
} // namespace dpl::test
