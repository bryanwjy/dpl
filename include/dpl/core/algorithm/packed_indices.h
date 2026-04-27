// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/std/bit/bit_ceil.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/utility/exchange.h"
#  include "dpl/std/utility/unreachable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace seq {

// Deliberately not exported!
template <size_t N>
class packed_indices {
    static_assert(__DPL has_single_bit(N));
    static constexpr bool is_blob = (N * __DPL bit_width(N - 1) > 64);
    static constexpr auto index_mask = N - 1;
    template <size_t>
    friend class packed_indices;

public:
    static constexpr auto index_width = __DPL bit_width(N - 1);
    static constexpr auto width = N * index_width;
    using underlying_type DPL_NODEBUG = bit_type_t<__DPL bit_ceil(width)>;

    static consteval size_t size() noexcept { return N; }

    consteval packed_indices(zero_t) noexcept : value_{} {}

    consteval packed_indices() noexcept : value_{make_index_sequence<N>{}} {}

    template <integral T, T... Is>
    requires (sizeof...(Is) == N)
    consteval packed_indices(integer_sequence<T, Is...>) noexcept
        : packed_indices(Is...) {}

    template <integral... T>
    requires (sizeof...(T) == N)
    explicit consteval packed_indices(T... args) noexcept : value_{} {
        unsigned char const all[] = {static_cast<unsigned char>(args)...};
        for (auto idx = 0; idx < N; ++idx) {
            set(idx, all[idx]);
        }
    }

    consteval size_t operator[](size_t idx) const noexcept {
        idx *= index_width;
        if constexpr (is_blob) {
            auto const byte_idx = idx / char_bit_v;
            auto const offset = idx % char_bit_v;

            return static_cast<size_t>(
                (static_cast<unsigned char>(value_.data[byte_idx]) >> offset) &
                index_mask);
        } else {
            return static_cast<size_t>((value_ >> idx) & index_mask);
        }
    }

private:
    consteval void set(size_t idx, unsigned char val) noexcept {
        if (__DPL bit_width(val) > index_width) {
            __DPL unreachable();
        }

        idx *= index_width;
        if constexpr (is_blob) {
            static_assert(index_width < sizeof(unsigned char) * char_bit_v);
            auto const byte_idx = idx / char_bit_v;
            auto const offset = idx % char_bit_v;
            if (char_bit_v - offset <= index_width) {
                value_.data[byte_idx] |= (val << offset);
            } else {
                auto const tmp = static_cast<unsigned short>(val) << offset;
                static constexpr auto byte_mask = 0xFFu;
                value_.data[byte_idx] |=
                    static_cast<unsigned char>(tmp & byte_mask);
                value_.data[1 + byte_idx] |=
                    static_cast<unsigned char>((tmp >> char_bit_v) & byte_mask);
            }
        } else {
            value_ |= (static_cast<underlying_type>(val) << idx);
        }
    }

    underlying_type value_;
};

template <size_t N>
consteval packed_indices<N> rotate(packed_indices<N> val, size_t num) noexcept {
    constexpr auto index_mask = N - 1;
    num &= index_mask;
    if (num == 0) {
        return val;
    }

    unsigned char out[N] = {};
    for (auto i = 0zu; i < N; ++i) {
        auto const idx = (i + num) % N;
        out[idx] = static_cast<unsigned char>(val[i]);
    }

    return [&]<size_t... Is>(index_sequence<Is...>) {
        return packed_indices<N>(out[Is]...);
    }(make_index_sequence<N>{});
}

template <size_t N, auto V>
consteval packed_indices<N> rotate(
    immediate_mask<N, V> mask, packed_indices<N> val, size_t num) noexcept {
    if constexpr (all_of(mask)) {
        return seq::rotate(val, num);
    } else if constexpr (none_of(mask)) {
        return val;
    } else {
        constexpr auto active_count = __DPL datapar::popcount(mask);
        num %= active_count;
        if (num == 0) {
            return val;
        }

        unsigned char active[active_count] = {};
        for (auto i = 0zu, j = 0zu; i < N; ++i) {
            if (mask[i]) {
                active[j++] = i;
            }
        }

        return [&]<size_t... Is>(index_sequence<Is...>) {
            unsigned char out[N] = {val[Is]...};

            for (auto moved = 0zu, start = 0zu; moved < active_count; ++start) {
                for (auto idx = start, prev = out[active[idx]]; start != idx;
                    ++moved) {
                    auto const next = (idx + num) % active_count;
                    prev = __DPL exchange(out[active[next]], prev);
                    idx = next;
                }
            }

            return packed_indices<N>(out[Is]...);
        }(make_index_sequence<N>{});
    }
}

template <__DPL integral... T>
explicit packed_indices(T... args) -> packed_indices<sizeof...(T)>;
} // namespace seq

using seq::packed_indices;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
