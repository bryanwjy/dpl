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
#  include "dpl/std/utility/unreachable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <size_t N>
class packed_indices {
    static_assert(__DPL has_single_bit(N));
    static constexpr bool is_blob = (N * __DPL bit_width(N - 1) > 64);
    static constexpr auto index_mask = N - 1;
    template <size_t>
    friend class packed_indices;

public:
    static constexpr auto index_width = __DPL bit_width(N - 1);
    static constexpr auto width = N * index_width;
    using underlying_type = bit_type_t<__DPL bit_ceil(width)>;

    consteval packed_indices(zero_t) noexcept : value_{} {}

    /**
     * Default constructed with 'iota'
     */
    consteval packed_indices() noexcept : value_{} {
        for (auto idx = 0; idx < N; ++idx) {
            set(idx, idx);
        }
    }

    template <__DPL integral... T>
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

    template <size_t F>
    requires (F > 1)
    consteval auto repeat() const noexcept {
        packed_indices<N * F> out{};
        for (auto i = 0zu; i < N; ++i) {
            for (auto j = 0zu; j < F; ++j) {
                out.set(i * F + j, (*this)[i]);
            }
        }

        return out;
    }

    /**
     * Shifting lower index element to higher positions, wrapping around
     * the array
     */
    consteval packed_indices<N> rotate_right(size_t val) const noexcept {
        val &= index_mask;
        if (val == 0) {
            return *this;
        }

        packed_indices<N> out{};
        for (auto i = 0zu; i < N; ++i) {
            auto const idx = (i + val) % N;
            out.set(idx, (*this)[i]);
        }

        return out;
    }

    /**
     * Shifting higher index element to lower positions, wrapping around
     * the array
     */
    consteval packed_indices<N> rotate_left(size_t val) const noexcept {
        val &= index_mask;
        if (val == 0) {
            return *this;
        }

        packed_indices<N> out{};
        for (auto i = 0zu; i < N; ++i) {
            auto const idx = (i + N - val) % N;
            out.set(idx, (*this)[i]);
        }

        return out;
    }

    template <__DPL convertible_to<bit_type_t<N>> auto M>
    consteval packed_indices<N> rotate_right(
        size_t val, basic_immediate_mask<N, M> mask = {}) const noexcept {
        if constexpr (all_of(mask)) {
            return rotate_right(val);
        } else if constexpr (none_of(mask)) {
            return *this;
        } else {
            return masked_rotate(mask, static_cast<ptrdiff_t>(val));
        }
    }

    template <__DPL convertible_to<bit_type_t<N>> auto M>
    consteval packed_indices<N> rotate_left(
        size_t val, basic_immediate_mask<N, M> mask = {}) const noexcept {
        if constexpr (all_of(mask)) {
            return rotate_right(val);
        } else if constexpr (none_of(mask)) {
            return *this;
        } else {
            return masked_rotate(mask, -static_cast<ptrdiff_t>(val));
        }
    }

#if 0
    /**
     * Add to each element, taking the modulus if out of range
     */
    consteval packed_indices<N>& operator+=(size_t delta) noexcept {
        delta &= index_mask;
        for (auto i = 0zu; i < N; ++i) {
            auto const val = ((*this)[i] + delta) & index_mask;
            this->set(i, static_cast<unsigned char>(val));
        }
        return *this;
    }

    consteval packed_indices<N> operator++(int) noexcept {
        auto const copy{*this};
        *this += 1;
        return copy;
    }

    consteval packed_indices<N>& operator++() noexcept { return *this += 1; }

    /**
     * Subtract from each element, taking the modulus if out of range
     */
    consteval packed_indices<N>& operator-=(size_t delta) noexcept {
        delta &= index_mask;
        for (auto i = 0zu; i < N; ++i) {
            auto const val = ((*this)[i] + N - delta) & index_mask;
            this->set(i, static_cast<unsigned char>(val));
        }
        return *this;
    }

    consteval packed_indices<N> operator--(int) noexcept {
        auto const copy{*this};
        *this -= 1;
        return copy;
    }

    consteval packed_indices<N>& operator--() noexcept { return *this -= 1; }

    consteval packed_indices<N> operator-(size_t delta) const noexcept {
        packed_indices<N> out{*this};
        out -= delta;
        return out;
    }

    consteval packed_indices<N> operator+(size_t delta) const noexcept {
        packed_indices<N> out{*this};
        out += delta;
        return out;
    }
#endif

    explicit consteval operator underlying_type() noexcept
    requires (!is_blob)
    {
        return value_;
    }

    consteval underlying_type operator+() noexcept
    requires (!is_blob)
    {
        return value_;
    }

    explicit consteval operator underlying_type const&() const noexcept
    requires is_blob
    {
        return value_;
    }

    consteval underlying_type const& operator+() noexcept
    requires is_blob
    {
        return value_;
    }

    consteval size_t size() const noexcept { return N; }
    consteval size_t size_bytes() const noexcept { return width / char_bit_v; }

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

private:
    template <auto MaskBits>
    consteval packed_indices masked_rotate(this packed_indices self,
        basic_immediate_mask<N, MaskBits> mask, ptrdiff_t shift) noexcept {
        auto const ones = datapar::popcount(mask);
        shift = shift < 0 ? ones - (-shift % ones) : shift % ones;
        auto const zero_mask = [&]() {
            packed_indices output;
            for (auto i = 0u; i < N; ++i) {
                if (!mask[i]) {
                    output.set(i, index_mask);
                }
            }
            return output;
        }();

        auto const movable = [&]() {
            packed_indices output;
            for (auto i = 0u, j = 0u; i < N; ++i) {
                if (mask[i]) {
                    output.set(j++, i);
                }
            }

            return output;
        }();

        packed_indices output;
        for (auto i = 0u; i < ones; ++i) {
            unsigned char const idx = movable[i];
            unsigned char const value = movable[(i + ones - shift) % ones];
            output.set(idx, value);
        }

        for (auto i = 0u; i < N; ++i) {
            output.set(i, self[i] & zero_mask[i]);
        }

        return output;
    }

    underlying_type value_;
};

template <__DPL integral... T>
explicit packed_indices(T... args) -> packed_indices<sizeof...(T)>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
