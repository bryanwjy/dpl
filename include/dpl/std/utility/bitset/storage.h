// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/has_single_bit.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace details::bitset {
consteval size_t ceil_pow2(size_t val) noexcept {
    return 1zu << (__DPL bit_width(val) - __DPL has_single_bit(val));
}

template <size_t W>
struct storage {
    using underlying_type = size_t[W / (sizeof(size_t) * char_bit_v) +
        (W % (sizeof(size_t) * char_bit_v) != 0)];
    underlying_type storage_;
};

template <size_t W>
requires requires { typename bit_type_t<bitset::ceil_pow2(W)>; }
struct storage<W> {
    using underlying_type = bit_type_t<details::bitset::ceil_pow2(W)>;
    underlying_type value_;
};

} // namespace details::bitset

DPL_DEFAULT_NAMESPACE_END
