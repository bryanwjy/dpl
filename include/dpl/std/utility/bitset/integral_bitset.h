// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/to_unsigned.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/details/bitset.h"
#  include "dpl/std/type_traits/extent.h"
#  include "dpl/std/type_traits/is_scalar.h"
#  include "dpl/std/type_traits/sequence.h"
#endif

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t W>
class alignas(W / __DPL char_bit_v) bitset;

template <size_t W>
requires requires { typename bit_type_t<details::utility::ceil_pow2(W)>; }
class bitset<W> : public details::utility::bitset_storage<W> {
    // TODO iterators?
    using base_type DPL_NODEBUG = details::utility::bitset_storage<W>;
    using base_type::value_;

public:
    static constexpr auto width = W;
    using typename base_type::underlying_type;

private:
    static constexpr underlying_type one = static_cast<underlying_type>(1);

    template <size_t>
    friend class bitset;

public:
    static constexpr underlying_type all = []() {
        if constexpr (W == sizeof(underlying_type) * char_bit_v) {
            return static_cast<underlying_type>(
                ~static_cast<underlying_type>(0));
        } else {
            return static_cast<underlying_type>((one << W) - one);
        }
    }();
    __DPL_HIDE_FROM_ABI constexpr bitset() noexcept : base_type{} {}

    template <integral T = underlying_type>
    __DPL_HIDE_FROM_ABI explicit(signed_integral<T> ||
        sizeof(T) != sizeof(underlying_type)) constexpr bitset(T val) noexcept
        : base_type([val]() {
            if constexpr (sizeof(T) * char_bit_v != W) {
                return static_cast<underlying_type>(val & all);
            } else {
                return static_cast<underlying_type>(val);
            }
        }()) {}

    template <same_as<bool>... Bs>
    requires (sizeof...(Bs) > 0) && (sizeof...(Bs) <= W)
    __DPL_HIDE_FROM_ABI explicit(sizeof...(Bs) != W) constexpr bitset(
        Bs... vals) noexcept
        : base_type([&]<size_t... Is>(index_sequence<Is...>) {
            return static_cast<underlying_type>((... |
                static_cast<underlying_type>(
                    static_cast<underlying_type>(vals) << Is)));
        }(make_index_sequence<sizeof...(Bs)>{})) {}

    __DPL_HIDE_FROM_ABI explicit constexpr operator underlying_type(
        this bitset self) noexcept {
        return self.value_;
    }

    template <size_t... Ws>
    requires (sizeof...(Ws) > 1 && (... + Ws) == W)
    __DPL_HIDE_FROM_ABI constexpr bitset(bitset<Ws>... vals) noexcept
        : base_type{} {
        [&]<size_t H, size_t... Ts>(this auto self, bitset<H> const& head,
            bitset<Ts> const&... tail) constexpr {
            constexpr auto args = sizeof...(tail) + 1zu;
            if constexpr (args > 1) {
                *this <<= self(tail...);
            }

            *this |= head;
            if constexpr (args < sizeof...(Ws)) {
                return H;
            }
        }(vals...);
    }

    template <size_t ToW>
    requires (ToW != W)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) explicit(ToW < W) constexpr
    operator bitset<ToW>() const noexcept {
        using To = typename bitset<ToW>::underlying_type;
        if constexpr (is_scalar_v<To>) {
            return bitset<ToW>(static_cast<To>(value_));
        } else {
            bitset<ToW> output;
            output |= *this;
            return output;
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) static constexpr size_t size() noexcept {
        return W;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool operator[](
        this bitset self, size_t idx) noexcept {
        constexpr underlying_type one = 1;
        return self.value_ & (one << idx);
    }

    template <size_t OW>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, PURE) constexpr auto operator&(
        bitset<OW> const& other) const noexcept {
        if constexpr (OW > W) {
            auto dst(other);
            dst &= *this;
            return dst;
        } else {
            auto dst(*this);
            dst &= other;
            return dst;
        }
    }

    template <size_t OW>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, PURE) constexpr auto operator|(
        bitset<OW> const& other) const noexcept {
        if constexpr (OW > W) {
            auto dst(other);
            dst |= *this;
            return dst;
        } else {
            auto dst(*this);
            dst |= other;
            return dst;
        }
    }

    template <size_t OW>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, PURE) constexpr auto operator^(
        bitset<OW> const& other) const noexcept {
        if constexpr (OW > W) {
            auto dst(other);
            dst ^= *this;
            return dst;
        } else {
            auto dst(*this);
            dst ^= other;
            return dst;
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset operator~(
        this bitset self) noexcept {
        if constexpr (W == sizeof(underlying_type) * char_bit_v) {
            return bitset(static_cast<underlying_type>(~self.value_));
        } else {
            return bitset(static_cast<underlying_type>(self.value_ ^ all));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset operator<<(
        this bitset self, size_t shift) noexcept {
        return bitset(self.value_ << shift);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset operator>>(
        this bitset self, size_t shift) noexcept {
        return bitset(static_cast<underlying_type>(self.value_ >> shift));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool test(
        this bitset self, size_t idx) noexcept {
        return self[idx];
    }

    template <size_t W2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool operator==(
        bitset<W2> const& other) const noexcept {
        if constexpr (integral<typename bitset<W2>::underlying_type>) {
            return this->value_ == other.value_;
        } else {
            return this->value_ == other.storage_[0] && [&]() {
                for (auto i = 1zu;
                    i < extent_v<typename bitset<W2>::underlying_type>; ++i) {
                    if (0zu != other.storage_[i]) {
                        return false;
                    }
                }

                return true;
            }();
        }
    }

    template <size_t W2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool operator!=(
        bitset<W2> const& other) const noexcept {
        return !(*this == other);
    }

    __DPL_HIDE_FROM_ABI constexpr void set(size_t idx) noexcept {
        this->value_ |= (one << idx);
    }

    __DPL_HIDE_FROM_ABI constexpr void set(size_t idx, bool val) noexcept {
        if (val) {
            set(idx);
        } else {
            clear(idx);
        }
    }

    __DPL_HIDE_FROM_ABI constexpr void clear(size_t idx) noexcept {
        this->value_ &= ~(one << idx);
    }

    __DPL_HIDE_FROM_ABI constexpr void clear() noexcept { this->value_ = 0; }

    __DPL_HIDE_FROM_ABI constexpr bitset& invert() noexcept {
        if constexpr (all == static_cast<underlying_type>(-1)) {
            this->value_ = ~this->value_;
        } else {
            this->value_ ^= all;
        }
        return *this;
    }

    template <size_t OW>
    requires (OW <= W)
    __DPL_HIDE_FROM_ABI constexpr bitset& operator&=(
        bitset<OW> const& other) noexcept {
        this->value_ &= other.value_;
        return *this;
    }

    template <size_t OW>
    requires (OW <= W)
    __DPL_HIDE_FROM_ABI constexpr bitset& operator|=(
        bitset<OW> const& other) noexcept {
        this->value_ |= other.value_;
        return *this;
    }

    template <size_t OW>
    requires (OW <= W)
    __DPL_HIDE_FROM_ABI constexpr bitset& operator^=(
        bitset<OW> const& other) noexcept {
        this->value_ ^= other.value_;
        return *this;
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator<<=(size_t shift) noexcept {
        if constexpr (__DPL has_single_bit(W)) {
            this->value_ <<= shift;
        } else {
            this->value_ = (this->value_ << shift) & all;
        }

        return *this;
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator>>=(size_t shift) noexcept {
        this->value_ >>= shift;
        return *this;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, REINITIALIZES) constexpr bitset&
    reinitialize() noexcept {
        if constexpr (__DPL has_single_bit(W)) {
            return *this;
        } else {
            this->value_ &= bitset<W>::all;
            return *this;
        }
    }
};

template <integral T>
bitset(T val) -> bitset<sizeof(T) * char_bit_v>;

template <integral_constant_like T>
explicit bitset(T val)
    -> bitset<__DPL bit_width(__DPL to_unsigned(T::value))>;

__DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
