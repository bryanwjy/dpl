// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.test.support:bitset_helpers;
import dpl;

export namespace dpl::test {
inline namespace support {
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

template <typename T>
using make_bitset_t = dpl::bitset<dpl::type_bit_v<T>>;

template <typename T>
[[nodiscard]] constexpr make_bitset_t<T> to_bitset(T const& data) noexcept
requires requires { dpl::bit_cast<make_bitset_t<T>>(data); }
{
    return dpl::bit_cast<make_bitset_t<T>>(data);
}
} // namespace support
} // namespace dpl::test
