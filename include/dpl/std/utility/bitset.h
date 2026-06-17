// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/apply.h"
#include "dpl/std/utility/sequence.h"
#include "dpl/std/utility/structured_bindings.h"
#include "dpl/std/utility/to_unsigned.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/byteswap.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/countl.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/type_traits/extent.h"
#  include "dpl/std/type_traits/is_base_of.h"
#  include "dpl/std/type_traits/is_scalar.h"
#endif

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::bitset {
consteval size_t ceil_pow2(size_t val) noexcept {
    return 1zu << (__DPL bit_width(val) - __DPL has_single_bit(val));
}

struct bypass_t {
    explicit consteval bypass_t() noexcept = default;
};
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

DPL_EXPORT template <size_t W>
class alignas(W / __DPL char_bit_v) bitset;

DPL_EXPORT template <size_t W>
requires requires { typename bit_type_t<details::bitset::ceil_pow2(W)>; }
class bitset<W> : private details::bitset::storage<W> {
    // TODO iterators?
    using base_type DPL_NODEBUG = details::bitset::storage<W>;
    using base_type::value_;

public:
    static constexpr auto width = W;
    using typename base_type::underlying_type;

private:
    static constexpr underlying_type one = static_cast<underlying_type>(1);
    static constexpr underlying_type all =
        static_cast<underlying_type>((one << W) - one);

    template <size_t>
    friend class bitset;
    using bypass_t = details::bitset::bypass_t;
    static constexpr bypass_t bypass{};

    __DPL_HIDE_FROM_ABI explicit constexpr bitset(
        bypass_t, underlying_type val) noexcept
        : base_type(val) {}

public:
    __DPL_HIDE_FROM_ABI constexpr bitset() noexcept : base_type{} {}

    template <integral T>
    __DPL_HIDE_FROM_ABI explicit(sizeof(T) * char_bit_v != W) constexpr bitset(
        T val) noexcept
        : base_type([val]() {
            if constexpr (sizeof(T) * char_bit_v != W) {
                return val & all;
            } else {
                return val;
            }
        }()) {}

    template <same_as<bool>... Bs>
    requires (sizeof...(Bs) > 0) && (sizeof...(Bs) <= W)
    __DPL_HIDE_FROM_ABI explicit(sizeof...(Bs) != W) constexpr bitset(
        Bs... vals) noexcept
        : base_type([&]<size_t... Is>(index_sequence<Is...>) {
            return (... | (vals << Is));
        }(make_index_sequence<sizeof...(Bs)>{})) {}

    __DPL_HIDE_FROM_ABI explicit constexpr operator underlying_type(
        this bitset self) noexcept {
        return self.value_;
    }

    template <size_t... Ws>
    requires (sizeof...(Ws) > 1 && (... + Ws) == W)
    __DPL_HIDE_FROM_ABI constexpr bitset(bitset<Ws>... vals) noexcept {
        [&]<size_t H, size_t... Ts>(this auto self, bitset<H> const& head,
            bitset<Ts> const&... tail) constexpr {
            constexpr auto args = sizeof...(vals) + 1zu;
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
    operator bitset<ToW>() noexcept {
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

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset operator&(
        this bitset self, bitset other) noexcept {
        return bitset(bypass, self.value_ & other.value_);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset operator|(
        this bitset self, bitset other) noexcept {
        return bitset(bypass, self.value_ | other.value_);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset operator^(
        this bitset self, bitset other) noexcept {
        return bitset(bypass, self.value_ ^ other.value_);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset operator~(
        this bitset self) noexcept {
        if constexpr (__DPL has_single_bit(W) &&
            W == sizeof(underlying_type) * char_bit_v) {
            return bitset(bypass, ~self.value_);
        } else {
            return bitset(bypass, self.value_ ^ all);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset operator<<(
        this bitset self, size_t shift) noexcept {
        if constexpr (__DPL has_single_bit(W)) {
            return bitset(bypass, self.value_ << shift);
        } else {
            return bitset(self.value_ << shift);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bitset operator>>(
        this bitset self, size_t shift) noexcept {
        return bitset(bypass, self.value_ >> shift);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool test(
        this bitset self, size_t idx) noexcept {
        return self[idx];
    }

    template <size_t W2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool operator==(
        bitset<W2> const& other) const {
        if constexpr (integral<typename bitset<W2>::underlying_type>) {
            return this->value_ == other.value_;
        } else {
            return this->value_ == other.storage_[0] && [&]() {
                for (auto v : other.storage_) {
                    if (v != 0) {
                        return false;
                    }
                }

                return true;
            }();
        }
    }

    template <size_t W2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool operator!=(
        bitset<W2> const& other) const {
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

    __DPL_HIDE_FROM_ABI constexpr bitset& operator&=(bitset other) noexcept {
        this->value_ &= other.value_;
        return *this;
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator|=(bitset other) noexcept {
        this->value_ |= other.value_;
        return *this;
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator^=(bitset other) noexcept {
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

DPL_EXPORT template <size_t W>
class bitset : private details::bitset::storage<W> {
    static_assert(W > 0);
    using base_type DPL_NODEBUG = details::bitset::storage<W>;
    using base_type::storage_;

    static constexpr auto chunk_size = sizeof(size_t) * char_bit_v;
    template <size_t>
    friend class bitset;

public:
    static constexpr auto width = W;
    using typename base_type::underlying_type;

private:
    template <size_t>
    friend class bitset;
    using bypass_t = details::bitset::bypass_t;
    static constexpr bypass_t bypass{};

public:
    __DPL_HIDE_FROM_ABI constexpr bitset() noexcept : base_type{} {}

    template <same_as<bool>... Bs>
    requires (sizeof...(Bs) > 0) && (sizeof...(Bs) <= W)
    __DPL_HIDE_FROM_ABI explicit(sizeof...(Bs) != W) constexpr bitset(
        Bs... vals) noexcept {
        [&]<size_t... Is>(index_sequence<Is...>) {
            (..., (vals ? set(Is) : clear(Is)));
        }(make_index_sequence<sizeof...(Bs)>{});
    }

    template <size_t... Ws>
    requires (sizeof...(Ws) > 1 && (... + Ws) == W)
    __DPL_HIDE_FROM_ABI constexpr bitset(bitset<Ws> const&... vals) noexcept {
        [&]<size_t H, size_t... Ts>(this auto self, bitset<H> const& head,
            bitset<Ts> const&... tail) constexpr {
            constexpr auto args = sizeof...(vals) + 1zu;
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
    operator bitset<ToW>() noexcept {
        using To = typename bitset<ToW>::underlying_type;
        if constexpr (ToW > W) {
            bitset<ToW> output;
            output |= *this;
            return output;
        } else if constexpr (integral<To>) {
            return bitset<ToW>(this->storage_[0]);
        } else {
            bitset<ToW> output;
            for (auto i = 0zu; i < extent_v<To>; ++i) {
                output.storage_[i] = this->storage_[i];
            }
            output.reinitialize();
            return output;
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) static constexpr size_t size() noexcept {
        return W;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool operator[](
        size_t idx) const noexcept {
        return this->storage_[idx / chunk_size] & (1zu << (idx % chunk_size));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool test(
        size_t idx) const noexcept {
        return (*this)[idx];
    }

    __DPL_HIDE_FROM_ABI constexpr void set(size_t idx) noexcept {
        auto ptr = this->storage_ + idx / chunk_size;
        *ptr |= (1zu << (idx % chunk_size));
    }

    __DPL_HIDE_FROM_ABI constexpr void clear(size_t idx) noexcept {
        auto ptr = this->storage_ + idx / chunk_size;
        *ptr &= ~(1zu << (idx % chunk_size));
    }

    __DPL_HIDE_FROM_ABI constexpr void set(size_t idx, bool val) noexcept {
        if (val) {
            set(idx);
        } else {
            clear(idx);
        }
    }

    template <size_t W2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool operator==(
        bitset<W2> const& other) const {
        if constexpr (integral<typename bitset<W2>::underlying_type>) {
            return other == *this;
        } else {
            constexpr auto size = extent_v<underlying_type> <
                    extent_v<typename bitset<W2>::underlying_type>
                ? extent_v<underlying_type>
                : extent_v<typename bitset<W2>::underlying_type>;
            for (auto i = 0zu; i < size; ++i) {
                if (this->storage_[i] != other.storage_[i]) {
                    return false;
                }
            }
            if constexpr (size == extent_v<underlying_type>) {
                for (auto i = size;
                    i < extent_v<typename bitset<W2>::underlying_type>; ++i) {
                    if (other.storage_[i] != 0) {
                        return false;
                    }
                }
            } else {
                for (auto i = size; i < extent_v<underlying_type>; ++i) {
                    if (this->storage_[i] != 0) {
                        return false;
                    }
                }
            }

            return true;
        }
    }

    template <size_t W2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool operator!=(
        bitset<W2> const& other) const {
        return !(*this == other);
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& invert() noexcept {
        for (auto& val : this->storage_) {
            val = ~val;
        }
        return this->reinitialize();
    }

    __DPL_HIDE_FROM_ABI constexpr void clear() noexcept {
        for (auto& val : this->storage_) {
            val = 0zu;
        }
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

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, PURE) constexpr bitset
    operator~() const noexcept {
        return auto(*this).invert();
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, PURE) constexpr bitset operator<<(
        size_t shift) const noexcept {
        auto dst(*this);
        dst <<= shift;
        return dst;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, PURE) constexpr bitset operator>>(
        size_t shift) noexcept {
        auto dst(*this);
        dst >>= shift;
        return dst;
    }

    template <size_t OW>
    requires (OW <= W)
    __DPL_HIDE_FROM_ABI constexpr bitset& operator&=(
        bitset<OW> const& other) noexcept {
        if constexpr (OW <= chunk_size) {
            this->storage_[0] &= other.value_;
        } else {
            auto* ptr = this->storage_;
            for (auto const val : other.storage_) {
                *ptr++ &= val;
            }

            for (auto const end = this->storage_ + extent_v<underlying_type>;
                ptr < end;) {
                *ptr++ = 0zu;
            }
        }

        return *this;
    }

    template <size_t OW>
    requires (OW <= W)
    __DPL_HIDE_FROM_ABI constexpr bitset& operator|=(
        bitset<OW> const& other) noexcept {
        using underlying = typename bitset<OW>::underlying_type;
        if constexpr (OW <= chunk_size) {
            this->storage_[0] |= other.value_;
        } else {
            for (auto* ptr = this->storage_; auto const val : other.storage_) {
                *ptr++ |= val;
            }
        }

        return *this;
    }

    template <size_t OW>
    requires (OW <= W)
    __DPL_HIDE_FROM_ABI constexpr bitset& operator^=(
        bitset<OW> const& other) noexcept {
        using underlying = typename bitset<OW>::underlying_type;
        if constexpr (OW <= chunk_size) {
            this->storage_[0] ^= other.value_;
        } else {
            for (auto* ptr = this->storage_; auto const val : other.storage_) {
                *ptr++ ^= val;
            }
        }

        return *this;
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator<<=(size_t shift) noexcept {

        if (shift >= W) {
            clear();
            return *this;
        }

        size_t const word_shift = shift / chunk_size;
        size_t const bit_shift = shift % chunk_size;

        constexpr size_t chunks = extent_v<underlying_type>;
        if (word_shift != 0) {
            for (auto i = chunks; i-- > word_shift;) {
                this->storage_[i] = this->storage_[i - word_shift];
            }

            for (auto i = 0zu; i < word_shift; ++i) {
                this->storage_[i] = 0;
            }
        }

        if (bit_shift != 0) {
            for (auto i = chunks - 1; i > 0; --i) {
                this->storage_[i] = (this->storage_[i] << bit_shift) |
                    (this->storage_[i - 1] >> (chunk_size - bit_shift));
            }

            this->storage_[0] <<= bit_shift;
        }

        return this->reinitialize();
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator>>=(size_t shift) noexcept {

        if (shift >= W) {
            clear();
            return *this;
        }

        auto const word_shift = shift / chunk_size;
        auto const bit_shift = shift % chunk_size;

        constexpr auto chunks = extent_v<underlying_type>;
        if (word_shift != 0) {
            for (auto i = 0zu; i + word_shift < chunks; ++i) {
                this->storage_[i] = this->storage_[i + word_shift];
            }

            for (auto i = chunks - word_shift; i < chunks; ++i) {
                this->storage_[i] = 0;
            }
        }

        if (bit_shift != 0) {
            for (auto i = 0zu; i + 1 < chunks; ++i) {
                this->storage_[i] = (this->storage_[i] >> bit_shift) |
                    (this->storage_[i + 1] << (chunk_size - bit_shift));
            }

            this->storage_[chunks - 1] >>= bit_shift;
        }

        return *this;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, REINITIALIZES) constexpr bitset&
    reinitialize() noexcept {
        constexpr auto tail_size = W % chunk_size;
        if constexpr (tail_size > 0) {
            constexpr size_t mask = ~(-1zu << tail_size);
            this->storage_[extent_v<underlying_type> - 1] &= mask;
        }

        return *this;
    }
};

template <same_as<bool>... Bs>
requires (sizeof...(Bs) > 0)
bitset(Bs... vals) -> bitset<sizeof...(Bs)>;

template <size_t... Ws>
bitset(bitset<Ws> const&... vals) -> bitset<(... + Ws)>;

DPL_EXPORT template <size_t W>
struct tuple_size<bitset<W>> : size_constant<W> {};

DPL_EXPORT template <size_t I, size_t W>
struct tuple_element<I, bitset<W>> {
    using type DPL_NODEBUG = bool;
};

DPL_EXPORT template <size_t I, size_t W>
requires requires { typename bitset<W>; } &&
    is_scalar_v<typename bitset<W>::underlying_type>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto get(bitset<W> value) noexcept {
    return value[I];
}

DPL_EXPORT template <size_t I, size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto get(
    bitset<W> const& value) noexcept {
    return value[I];
}

template <typename T>
inline constexpr bool is_bitset_v = false;
template <typename T>
inline constexpr bool is_bitset_v<T const> = is_bitset_v<T>;
template <typename T>
inline constexpr bool is_bitset_v<T volatile> = is_bitset_v<T>;
template <typename T>
inline constexpr bool is_bitset_v<T const volatile> = is_bitset_v<T>;
template <size_t W>
inline constexpr bool is_bitset_v<bitset<W>> = true;

template <typename T>
concept bitset_type = is_bitset_v<T>;

template <typename T>
concept integral_bitset_type =
    bitset_type<T> && integral<typename T::underlying_type>;

DPL_EXPORT template <typename T>
concept bitset_constant_like = requires { T::value; } &&
    bitset_type<decltype(T::value)> && convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> &&
    bool_constant<T() == T::value>::value &&
    bool_constant<static_cast<decltype(T::value)>(T()) == T::value>::value;

template <bitset_type auto T>
using bitset_constant = integral_constant<decltype(T), T>;

DPL_EXPORT template <size_t W>
requires integral_bitset_type<bitset<W>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto to_underlying(
    bitset<W> val) noexcept {
    return __DPL bit_cast<typename bitset<W>::underlying_type>(val);
}

template <size_t W, size_t W2>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bitset<W> truncate(bitset<W2> const& val) noexcept {
    static_assert(W <= W2);
    return static_cast<bitset<W>>(val);
}

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int popcount(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        return __DPL popcount(__DPL to_underlying(val));
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        auto const& base = (details::bitset::storage<W> const&)val;
        auto sum = 0zu;
        for (auto const val : base.storage_) {
            sum += __DPL popcount(val);
        }

        return sum;
    }
}

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countl_one(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        constexpr auto chunk = sizeof(bitset<W>) * __DPL char_bit_v;
        constexpr auto padding = chunk - W;
        if constexpr (padding > 0) {
            return __DPL countl_one(
                __DPL to_unsigned(__DPL to_underlying(val) << padding));
        } else {
            return __DPL countl_one(__DPL to_underlying(val));
        }
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;

        auto const& base = (details::bitset::storage<W> const&)val;
        auto const* ptr = base.storage_ + extent_v<type>;

        auto result = 0zu;
        if constexpr (tail_size > 0) {
            auto const count = __DPL countl_one(*--ptr << padding);
            if (count < tail_size) {
                return count;
            }
        }

        for (; ptr > base.storage_; ++result) {
            if (auto const val = *--ptr; val != -1zu) {
                return result * chunk + tail_size + __DPL countl_one(val);
            }
        }

        return W;
    }
}

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countl_zero(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        constexpr auto chunk = sizeof(bitset<W>) * __DPL char_bit_v;
        constexpr auto padding = chunk - W;
        if constexpr (padding > 0) {
            return __DPL countl_zero(
                __DPL to_unsigned( __DPL to_underlying(val) << padding));
        } else {
            return __DPL countl_zero(__DPL to_underlying(val));
        }
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;

        auto const& base = (details::bitset::storage<W> const&)val;
        auto const* ptr = base.storage_ + extent_v<type>;

        auto result = 0zu;
        if constexpr (tail_size > 0) {
            auto const count = __DPL countl_zero(*--ptr << padding);
            if (count < tail_size) {
                return count;
            }
        }

        for (; ptr > base.storage_; ++result) {
            if (auto const val = *--ptr; val != 0zu) {
                return result * chunk + tail_size + __DPL countl_zero(val);
            }
        }

        return W;
    }
}

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countr_one(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        return __DPL countr_one(__DPL to_underlying(val));
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;
        auto const& base = (details::bitset::storage<W> const&)val;
        for (auto result = 0zu; auto const val : base.storage_) {
            if (val != -1zu) {
                return result * chunk + __DPL countr_one(val);
            }

            ++result;
        }

        return W;
    }
}

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countr_zero(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        return __DPL countr_one(__DPL to_underlying(val));
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;
        auto const& base = (details::bitset::storage<W> const&)val;
        for (auto result = 0zu; auto const val : base.storage_) {
            if (val != 0zu) {
                return result * chunk + __DPL countr_zero(val);
            }

            ++result;
        }

        return W;
    }
}

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bitset<W> byteswap(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        constexpr auto chunk = sizeof(bitset<W>) * __DPL char_bit_v;
        constexpr auto padding = chunk - W;
        if constexpr (padding > 0) {
            return bitset<W>( __DPL byteswap(
                __DPL to_unsigned(__DPL to_underlying(val) << padding)));
        } else {
            return bitset<W>(__DPL byteswap(__DPL to_underlying(val)));
        }
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;
        auto ret = __DPL bit_cast<bitset<W + padding>>(val);
        ret <<= padding;
        auto& base = (details::bitset::storage<W + padding>&)ret;
        for (auto left = base.storage_,
                  right = base.storage_ + extent_v<type> - 1;
            left < right;) {
            auto const tmp = *left;
            *left++ = __DPL byteswap(*right);
            *right-- = __DPL byteswap(tmp);
        }

        return __DPL bit_cast<bitset<W>>(ret);
    }
}

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bitset<W> rotl(bitset<W> const& val, int count) noexcept {
    // TODO: optimize for large sets
    count %= W;
    if (count == 0)
        return val;

    if (count > 0) {
        return (val << count) | (val >> (W - count));
    }

    return (val >> -count) | (val << (W + count));
}

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bitset<W> rotr(bitset<W> const& val, int count) noexcept {
    // TODO: optimize for large sets
    count %= W;
    if (count == 0)
        return val;

    if (count > 0) {
        return (val >> count) | (val << (W - count));
    }

    return (val << -count) | (val >> (W + count));
}

namespace internal::bitset {
// Not exported
template <size_t W>
struct bitstr {
    char data[W + 1];

    static consteval size_t size() noexcept { return W; }
    consteval char const* begin() const noexcept { return data + W; }
    consteval char const* end() const noexcept { return data + 2; }
    consteval unsigned radix() const {
        char prefix = data[1];
        bool is_bin = (prefix == 'b' || prefix == 'B');
        bool is_hex = (prefix == 'x' || prefix == 'X');
        return is_bin ? 2 : is_hex ? 16 : 8;
    }
};

} // namespace internal::bitset

namespace bit_literals {

DPL_EXPORT template <internal::bitset::bitstr S>
DPL_NODISCARD consteval auto operator""_bits() noexcept {
    bitset<S.size()> output;
    static_assert(S.data[0] == '0');
    static_assert([]() {
        constexpr auto radix = S.radix();
        if constexpr (radix == 2) {
            for (auto val : S.data) {
                if (val != '0' && val != '1') {
                    return false;
                }
            }
        } else if constexpr (radix == 16) {
            for (auto val : S.data) {
                if (!(val >= 'a' && val <= 'f') &&
                    !(val >= 'A' && val <= 'F')) {
                    return false;
                }
            }
        } else {
            for (auto val : S.data) {
                if (val < '0' || val > '7') {
                    return false;
                }
            }
        }
        return true;
    }());

    constexpr bool is_bin = S.radix() == 2;
    constexpr bool is_hex = S.radix() == 16;
    if constexpr (is_bin) {
        for (auto idx = 0zu; auto const val : S) {
            output.set(idx++, val == '1');
        }
    } else if constexpr (is_hex) {
        for (auto idx = 0zu; auto const val : S) {
            auto const set = bitset<4>((val >= '0' && val <= '9') ? val - '0'
                    : (val >= 'a' && val <= 'f') ? 10 + (val - 'a')
                                                 : 10 + (val - 'A'));
            __DPL apply(
                [&](auto... bs) { (..., output.set(idx++, bs)); }, set);
        }
    } else {
        for (auto idx = 0zu; auto const val : S) {
            auto const set = bitset<3>(val - '0');
            __DPL apply(
                [&](auto... bs) { (..., output.set(idx++, bs)); }, set);
        }
    }

    return output;
}

} // namespace bit_literals

DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
