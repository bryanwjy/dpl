#pragma once

#include "dpl/config.h"

#include "dpl/bit_type.h"
#include "dpl/blob.h"

#if !DPL_MODULES
#  include <bit>
#  include <climits>
#  include <concepts>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT_BEGIN

template <size_t N>
class packed_indices {
    static_assert(N && !(N & (N - 1)));
    static constexpr bool is_blob = (N * std::bit_width(N) > 64);
    static constexpr auto index_mask = (1zu << std::bit_width(N)) - 1;
    template <size_t>
    friend class packed_indices;

public:
    static constexpr auto index_width = std::bit_width(N);
    static constexpr auto width = N * index_width;
    using underlying_type = bit_type_t<std::bit_ceil(width)>;

    __DPL_HIDE_FROM_ABI consteval packed_indices() noexcept : value_{} {}

    template <std::integral... T>
    requires (sizeof...(T) == N)
    __DPL_HIDE_FROM_ABI explicit consteval packed_indices(T... args) noexcept
        : value_{} {
        unsigned char const all[] = {static_cast<unsigned char>(args)...};
        for (auto idx = 0; idx < N; ++idx) {
            set(idx, all[idx]);
        }
    }

    __DPL_HIDE_FROM_ABI explicit consteval packed_indices(
        bitset<N> mask, ptrdiff_t shift = 0) noexcept
        : packed_indices{masked_shift(mask, shift)} {}

    template <abi_tag Abi, simd_element T>
    requires (element_count<Abi, T> == N)
    __DPL_HIDE_FROM_ABI explicit consteval packed_indices(
        Abi, iota_t<T>) noexcept
        : value_{} {
        for (auto idx = 0; idx < N; ++idx) {
            set(idx, idx);
        }
    }

    __DPL_HIDE_FROM_ABI consteval unsigned char operator[](
        size_t idx) const noexcept {
        idx *= index_width;
        if constexpr (is_blob) {
            auto const byte_idx = idx / CHAR_BIT;
            auto const offset = idx % CHAR_BIT;

            return static_cast<unsigned char>(
                (static_cast<unsigned char>(value_.data[byte_idx]) >> offset) &
                index_mask);
        } else {
            return (value_ >> idx) & index_mask;
        }
    }

    template <size_t F>
    requires (F > 1)
    __DPL_HIDE_FROM_ABI consteval auto repeat() const noexcept {
        packed_indices<N * F> out{};
        for (auto i = 0zu; i < N; ++i) {
            for (auto j = 0zu; j < F; ++j) {
                out.set(i * F + j, *this[i]);
            }
        }

        return out;
    }

    /**
     * Shifting lower index element to higher positions, wrapping around
     * the array
     */
    __DPL_HIDE_FROM_ABI consteval packed_indices<N> operator>>(
        size_t val) const noexcept {
        val &= index_mask;
        if (val == 0) {
            return *this;
        }

        packed_indices<N> out{};
        for (auto i = 0zu; i < N; ++i) {
            auto const idx = (i + val) % N;
            out.set(idx, *this[i]);
        }

        return out;
    }

    /**
     * Shifting higher index element to lower positions, wrapping around
     * the array
     */
    __DPL_HIDE_FROM_ABI consteval packed_indices<N> operator<<(
        size_t val) const noexcept {
        val &= index_mask;
        if (val == 0) {
            return *this;
        }

        packed_indices<N> out{};
        for (auto i = 0zu; i < N; ++i) {
            auto const idx = (i + N - val) % N;
            out.set(idx, *this[i]);
        }

        return out;
    }

    /**
     * Add to each element, taking the modulus if out of range
     */
    __DPL_HIDE_FROM_ABI consteval packed_indices<N>& operator+=(
        size_t delta) noexcept {
        delta &= index_mask;
        for (auto i = 0zu; i < N; ++i) {
            auto const val = (*this[i] + delta) & index_mask;
            this->set(i, static_cast<unsigned char>(val));
        }
        return *this;
    }

    __DPL_HIDE_FROM_ABI consteval packed_indices<N> operator++(int) noexcept {
        auto const copy{*this};
        *this += 1;
        return copy;
    }

    __DPL_HIDE_FROM_ABI consteval packed_indices<N>& operator++() noexcept {
        return *this += 1;
    }

    /**
     * Subtract from each element, taking the modulus if out of range
     */
    __DPL_HIDE_FROM_ABI consteval packed_indices<N>& operator-=(
        size_t delta) noexcept {
        delta &= index_mask;
        for (auto i = 0zu; i < N; ++i) {
            auto const val = (*this[i] + N - delta) & index_mask;
            this->set(i, static_cast<unsigned char>(val));
        }
        return *this;
    }

    __DPL_HIDE_FROM_ABI consteval packed_indices<N> operator--(int) noexcept {
        auto const copy{*this};
        *this -= 1;
        return copy;
    }

    __DPL_HIDE_FROM_ABI consteval packed_indices<N>& operator--() noexcept {
        return *this -= 1;
    }

    __DPL_HIDE_FROM_ABI consteval packed_indices<N> operator-(
        size_t delta = 1) const noexcept {
        packed_indices<N> out{*this};
        out -= delta;
        return out;
    }

    __DPL_HIDE_FROM_ABI consteval packed_indices<N> operator+(
        size_t delta = 1) const noexcept {
        packed_indices<N> out{*this};
        out += delta;
        return out;
    }

    __DPL_HIDE_FROM_ABI explicit consteval operator underlying_type() noexcept
    requires (!is_blob)
    {
        return value_;
    }

    __DPL_HIDE_FROM_ABI consteval underlying_type operator+() noexcept
    requires (!is_blob)
    {
        return value_;
    }

    __DPL_HIDE_FROM_ABI explicit consteval underlying_type const&()
        const noexcept
    requires is_blob
    {
        return value_;
    }

    __DPL_HIDE_FROM_ABI consteval underlying_type const& operator+() noexcept
    requires is_blob
    {
        return value_;
    }

    __DPL_HIDE_FROM_ABI consteval size_t size() const noexcept { return N; }
    __DPL_HIDE_FROM_ABI consteval size_t size_bytes() const noexcept {
        return width / CHAR_BIT;
    }

    __DPL_HIDE_FROM_ABI void set(size_t idx, unsigned char val) noexcept {
        if (std::bit_width(val) >= index_width) {
            throw val;
        }

        idx *= index_width;
        if constexpr (is_blob) {
            static_assert(index_width < sizeof(std::byte) * CHAR_BIT);
            auto const byte_idx = idx / CHAR_BIT;
            auto const offset = idx % CHAR_BIT;
            if (CHAR_BIT - offset <= index_width) {
                value_.data[byte_idx] |= (val << offset);
            } else {
                auto const tmp = static_cast<unsigned short>(val) << offset;
                static constexpr auto byte_mask = 0xFFu;
                value_.data[byte_idx] |=
                    static_cast<std::byte>(tmp & byte_mask);
                value_.data[1 + byte_idx] |=
                    static_cast<std::byte>((tmp >> CHAR_BIT) & byte_mask);
            }
        } else {
            value_ |= (static_cast<underlying_type>(val) << idx);
        }
    }

private:
    __DPL_HIDE_FROM_ABI static consteval packed_indices masked_shift(
        bitset<N> mask, ptrdiff_t shift) noexcept {
        auto const ones = mask.popcount();
        shift = shift < 0 ? ones - (-shift % ones) : shift % ones;
        auto const iota = []() {
            packed_indices output;
            for (auto i = 0u; i < N; ++i) {
                output.set(i, i);
            }
            return output;
        }();

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
                    movable.set(j++, i);
                }
            }

            return output;
        };

        packed_indices output;
        for (auto i = 0u; i < ones; ++i) {
            unsigned char const idx = movable[i];
            unsigned char const value = movable[(i + ones - shift) % ones];
            output.set(idx, value);
        }

        for (auto i = 0u; i < N; ++i) {
            output.set(i, iota[i] & zero_mask[i]);
        }

        return output;
    }

    underlying_type value_;
};

template <std::integral... T>
explicit packed_indices(T... args) -> packed_indices<sizeof...(T)>;
template <size_t N>
explicit packed_indices(bitset<N> mask, ptrdiff_t shift) -> packed_indices<N>;
template <size_t N>
explicit packed_indices(bitset<N> mask) -> packed_indices<N>;
template <abi_tag Abi, simd_element T>
explicit packed_indices(Abi, iota_t<T>)
    -> packed_indices<element_count<Abi, T>>;

DPL_EXPORT_END

DPL_DEFAULT_NAMESPACE_END
