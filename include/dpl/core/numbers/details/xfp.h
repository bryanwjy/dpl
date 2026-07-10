// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT namespace details::numbers {
template <size_t N>
struct xfp {
    static_assert(80 % __DPL char_bit_v == 0);
    static constexpr auto bytes = 80zu / __DPL char_bit_v;
    struct val_t {
        unsigned char data[bytes];
    } val;
    struct padding_t {
        unsigned char data[N - bytes];
    } padding;
    consteval bitset<80> to_bitset() const {
        xfp copy{};
        copy.val = this->val;
        return __DPL bit_cast<bitset<80>>(copy);
    }
};
} // namespace details::numbers
DPL_DEFAULT_NAMESPACE_END
