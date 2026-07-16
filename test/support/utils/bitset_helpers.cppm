// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.test:utils.bitset_helpers;
import dpl;

namespace dpl::test {

template <size_t N>
[[nodiscard]] constexpr auto repeat_byte(unsigned char val) noexcept {
    dpl::bitset<dpl::char_bit_v> const byte(val);
    constexpr auto bytes = N / dpl::char_bit_v + (N % dpl::char_bit_v > 0);
    return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
        auto repeated = dpl::truncate<N>(dpl::bitset(((void)Is, byte)...));
        if constexpr (requires { dpl::to_underlying(repeated); }) {
            return dpl::to_underlying(repeated);
        } else {
            return repeated;
        }
    }(dpl::make_index_sequence<bytes>{});
}

} // namespace dpl::test
