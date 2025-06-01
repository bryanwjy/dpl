#pragma once

#include "dpl/config.h"

#include "dpl/blob.h"
#include "dpl/concepts.h"
#include "dpl/type_traits.h"

#if !DPL_MODULES
#  include <concepts>
#  include <cstddef>
#  include <type_traits>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t N>
class bitset {
    static_assert(N && !(N & (N - 1)));
    static constexpr bool is_blob = (N > 64);
    static constexpr auto total_bytes = sizeof(underlying_type);
    static_assert(total_bytes * CHAR_BIT == N);
    template <size_t>
    friend class bitset;

public:
    using underlying_type = bit_type_t<N>;

    static constexpr auto width = N;

    __DPL_HIDE_FROM_ABI consteval bitset() noexcept : value_{} {}
    template <std::convertible_to<std::byte>... T>
    requires std::semiregular<T> && (sizeof...(T) == total_bytes) && is_blob
    __DPL_HIDE_FROM_ABI explicit consteval bitset(T... args) noexcept
        : value_{static_cast<std::byte>(args)...} {}
    __DPL_HIDE_FROM_ABI explicit consteval bitset(underlying_type val) noexcept
        : value_{val} {}

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
        return total_bytes;
    }

    __DPL_HIDE_FROM_ABI consteval bool operator[](size_t idx) const noexcept {
        if constexpr (std::is_array_v<underlying_type>) {
            auto const byte_idx = idx / CHAR_BIT;
            auto const bit_idx = idx % CHAR_BIT;
            return (value_[byte_idx] >> bit_idx) & 1;
        } else {
            return (value_ >> idx) & 1;
        }
    }

    __DPL_HIDE_FROM_ABI consteval bitset operator&(
        bitset const& other) const noexcept {
        if constexpr (std::is_array_v<underlying_type>) {
            bitset out;
            for (auto& val : out) {
                val = value_ & other.value_;
            }
            return out;
        } else {
            return bitset{value_ & other.value_};
        }
    }

    __DPL_HIDE_FROM_ABI consteval bitset operator|(
        bitset const& other) const noexcept {
        if constexpr (std::is_array_v<underlying_type>) {
            bitset out;
            for (auto& val : out) {
                val = value_ | other.value_;
            }
            return out;
        } else {
            return bitset{value_ | other.value_};
        }
    }

    __DPL_HIDE_FROM_ABI consteval bitset operator~() const noexcept {
        if constexpr (std::is_array_v<underlying_type>) {
            bitset out{};
            for (auto i = 0uz; i < width; ++i) {
                out.value_[i] = ~value_[i];
            }
            return out;
        } else {
            return bitset{~value_};
        }
    }

    __DPL_HIDE_FROM_ABI consteval bitset operator<<(
        size_t shift) const noexcept {
        if constexpr (std::is_array_v<underlying_type>) {
            if (shift == 0) {
                return *this;
            }
            if (shift >= width) {
                return bitset{};
            }

            bitset out{};
            auto const byte_shift = shift / CHAR_BIT;
            auto const bit_shift = shift % CHAR_BIT;

            for (auto i = 0zu; i < total_bytes; ++i) {
                auto const src = i + byte_shift;
                unsigned short val = 0;
                if (src < total_bytes) {
                    val |= (value_[src] << bit_shift);
                }

                if (bit_shift && src + 1 < total_bytes) {
                    val |= (value_[src + 1] >> (CHAR_BIT - bit_shift));
                }

                out.value_[i] =
                    (src < total_bytes) ? (unsigned char)(val & 0xFF) : 0;
            }

            return out;
        } else {
            return bitset{value_ << shift};
        }
    }

    __DPL_HIDE_FROM_ABI consteval bitset operator>>(
        size_t shift) const noexcept {
        if constexpr (std::is_array_v<underlying_type>) {
            if (shift == 0) {
                return *this;
            }
            if (shift >= width) {
                return bitset{};
            }

            bitset out{};
            auto const byte_shift = shift / CHAR_BIT;
            auto const bit_shift = shift % CHAR_BIT;

            for (auto i = total_bytes - 1; i >= 0zu; --i) {
                auto const src = i - byte_shift;
                unsigned short val = 0;

                if (src >= 0) {
                    val |= (value_[src] >> bit_shift);
                }

                if (bit_shift && src - 1 >= 0) {
                    val |= (value_[src - 1] << (CHAR_BIT - bit_shift));
                }

                out.value_[i] = (src >= 0) ? (unsigned char)(val & 0xFF) : 0;
            }
            return out;
        } else {
            return bitset{value_ << shift};
        }
    }

    __DPL_HIDE_FROM_ABI consteval size_t popcount() const noexcept {
        if constexpr (is_blob) {
            auto count = 0zu;
            for (auto idx = 0zu; idx < total_bytes; ++idx) {
                count += std::popcount(value_.data[idx]);
            }
        } else {
            return std::popcount(value_);
        }
    }

    template <size_t F>
    requires (F > 1)
    __DPL_HIDE_FROM_ABI consteval auto repeat() const noexcept {
        bitset<width * F> out;
        for (auto i = 0zu; i < width; ++i) {
            if (*this[i]) {
                for (auto j = 0zu; j < F; ++j) {
                    out.set(i * F + j);
                }
            }
        }

        return out;
    }

    template <size_t F>
    requires (F > 1)
    __DPL_HIDE_FROM_ABI consteval bitset<M> zero_extend() const noexcept {
        bitset<width * F> out;
        for (auto i = 0zu; i < width; ++i) {
            if (*this[i]) {
                out.set(i * F);
            }
        }

        return out;
    }

    template <size_t F>
    requires (F > 1)
    __DPL_HIDE_FROM_ABI consteval bitset<M> reduce() const noexcept {
        bitset<width / F> out;
        for (auto i = 0zu; i < width; ++i) {
            if (*this[i * F + F - 1]) {
                out.set(i);
            }
        }

        return out;
    }

    template <size_t M>
    requires (M != width)
    __DPL_HIDE_FROM_ABI consteval bitset<M> truncate() const noexcept {
        bitset<M> out;
        for (auto i = 0zu; i < M && i < width; ++i) {
            if (*this[i]) {
                out.set(i);
            }
        }

        return out;
    }

    __DPL_HIDE_FROM_ABI consteval void set(size_t idx) noexcept {
        if constexpr (std::is_array_v<underlying_type>) {
            auto const byte_idx = idx / CHAR_BIT;
            auto const bit_idx = idx % CHAR_BIT;
            value_.data[byte_idx] |= (static_cast<unsigned char>(1) << bit_idx);
        } else {
            value_ |= (static_cast<underlying_type>(1) << idx);
        }
    }

    __DPL_HIDE_FROM_ABI consteval void clear(size_t idx) noexcept {
        if constexpr (std::is_array_v<underlying_type>) {
            auto const byte_idx = idx / CHAR_BIT;
            auto const bit_idx = idx % CHAR_BIT;
            value_.data[byte_idx] &=
                ~(static_cast<unsigned char>(1) << bit_idx);
        } else {
            value_ &= ~(static_cast<underlying_type>(1) << idx);
        }
    }

private:
    underlying_type value_;
};

DPL_DEFAULT_NAMESPACE_END
