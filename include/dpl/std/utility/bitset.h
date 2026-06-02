// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/apply.h"
#include "dpl/std/utility/ignore.h"
#include "dpl/std/utility/sequence.h"
#include "dpl/std/utility/structured_bindings.h"
#include "dpl/std/utility/to_unsigned.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/type_traits/extent.h"
#  include "dpl/std/type_traits/is_scalar.h"
#endif

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::bitset {
consteval size_t ceil_pow2(size_t val) noexcept {
    return 1zu << (__DPL bit_width(val) - 1);
}

struct bypass_t {
    explicit consteval bypass_t() noexcept = default;
};

} // namespace details::bitset

DPL_EXPORT template <size_t W>
class alignas(W / __DPL char_bit_v) bitset;

DPL_EXPORT template <size_t W>
requires requires { typename bit_type_t<details::bitset::ceil_pow2(W)>; }
class bitset<W> {
    // TODO iterators?
public:
    static constexpr auto width = W;
    using underlying_type = bit_type_t<details::bitset::ceil_pow2(W)>;

private:
    static constexpr underlying_type one = static_cast<underlying_type>(1);
    static constexpr underlying_type all = W == details::bitset::ceil_pow2(W)
        ? static_cast<underlying_type>(-1)
        : static_cast<underlying_type>((one << W) - one);

    template <size_t>
    friend class bitset;
    using bypass_t = details::bitset::bypass_t;
    static constexpr bypass_t bypass{};

    __DPL_HIDE_FROM_ABI explicit constexpr bitset(
        bypass_t, underlying_type val) noexcept
        : value_(val) {}

public:
    __DPL_HIDE_FROM_ABI constexpr bitset() noexcept : value_{} {}

    template <integral T>
    __DPL_HIDE_FROM_ABI explicit(sizeof(T) * char_bit_v != W) constexpr bitset(
        T val) noexcept
        : value_([val]() {
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
        : value_([&]<size_t... Is>(index_sequence<Is...>) {
            return (... | (vals << Is));
        }(make_index_sequence<sizeof...(Bs)>{})) {}

    __DPL_HIDE_FROM_ABI explicit constexpr operator underlying_type(
        this bitset self) noexcept {
        return self.value_;
    }

    template <size_t... Ws>
    requires (sizeof...(Ws) > 1 && (... + Ws) == W)
    __DPL_HIDE_FROM_ABI constexpr bitset(bitset<Ws>... vals) noexcept {
#if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && \
    __cpp_expansion_statements >= 202506L
        template for (auto offset = 0zu; auto const val : {vals...}) {
            value_ |= static_cast<underlying_type>(val.value_) << offset;
            offset += val.size();
        }
#else
        auto offset = 0zu;
        auto const append = [&]<size_t N>(
                                this auto self, bitset<N> v) constexpr {
            value_ |= static_cast<underlying_type>(v.value_) << offset;
            offset += v.size();
        };

        (..., append(vals));
#endif
    }

    template <size_t ToW>
    requires (ToW > W)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) explicit(
        !is_scalar_v<typename bitset<ToW>::underlying_type>) constexpr
    operator bitset<ToW>() noexcept {
        if constexpr (is_scalar_v<typename bitset<ToW>::underlying_type>) {
            return bitset<ToW>(bypass, value_);
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
        if constexpr (__DPL has_single_bit(W)) {
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

    __DPL_HIDE_FROM_ABI constexpr void set(size_t idx) noexcept {
        value_ |= (one << idx);
    }

    __DPL_HIDE_FROM_ABI constexpr void set(size_t idx, bool val) noexcept {
        if (val) {
            set(idx);
        } else {
            clear(idx);
        }
    }

    __DPL_HIDE_FROM_ABI constexpr void clear(size_t idx) noexcept {
        value_ &= ~(one << idx);
    }

    __DPL_HIDE_FROM_ABI constexpr void clear() noexcept { value_ = 0; }

    __DPL_HIDE_FROM_ABI constexpr bitset& invert() noexcept {
        if constexpr (all == static_cast<underlying_type>(-1)) {
            value_ = ~value_;
        } else {
            value_ ^= all;
        }
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator&=(bitset other) noexcept {
        value_ &= other.value_;
        return *this;
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator|=(bitset other) noexcept {
        value_ |= other.value_;
        return *this;
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator^=(bitset other) noexcept {
        value_ ^= other.value_;
        return *this;
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator<<=(size_t shift) noexcept {
        if constexpr (__DPL has_single_bit(W)) {
            value_ <<= shift;
        } else {
            value_ = (value_ << shift) & all;
        }

        return *this;
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& operator>>=(size_t shift) noexcept {
        value_ >>= shift;
        return *this;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, REINITIALIZES) constexpr bitset&
    reinitialize() noexcept {
        if constexpr (__DPL has_single_bit(W)) {
            return *this;
        } else {
            value_ &= bitset<W>::all;
            return *this;
        }
    }

private:
    underlying_type value_;
};

template <integral T>
bitset(T val) -> bitset<sizeof(T) * char_bit_v>;

template <integral_constant_like T>
explicit bitset(T val)
    -> bitset<__DPL bit_width(__DPL to_unsigned(T::value))>;

DPL_EXPORT template <size_t W>
class bitset {
    static_assert(W > 0);
    static constexpr auto chunk_size = sizeof(size_t) * char_bit_v;
    template <size_t>
    friend class bitset;

public:
    static constexpr auto width = W;
    using underlying_type =
        size_t[width / chunk_size + (width % chunk_size != 0)];

private:
    template <size_t>
    friend class bitset;
    using bypass_t = details::bitset::bypass_t;
    static constexpr bypass_t bypass{};
    static constexpr make_index_sequence<chunk_size> chunk_sequence{};
    static constexpr make_index_sequence<extent_v<underlying_type>>
        extent_sequence{};
    static constexpr make_index_sequence<W> full_sequence{};

public:
    __DPL_HIDE_FROM_ABI explicit constexpr bitset(integral auto val) noexcept
    requires (sizeof(val) < sizeof(size_t))
        : storage_(val) {}

    template <size_t InW>
    requires (InW < W)
    __DPL_HIDE_FROM_ABI explicit constexpr bitset(
        bitset<InW> const& vals) noexcept {
        *this |= vals;
    }

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
    __DPL_HIDE_FROM_ABI constexpr bitset(bitset<Ws>... vals) noexcept {
#if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && \
    __cpp_expansion_statements >= 202506L
        template for (auto offset = 0zu; auto const& val : {vals...}) {
            template for (idx = 0zu; auto v : val) {
                set(offset + idx, v.test(idx));
                ++idx;
            }
            offset += val.size();
        }
#else
        auto offset = 0zu;
        auto const append = [&]<size_t N>(bitset<N> v) constexpr {
            for (auto i = 0zu; i < v.size(); ++i) {
                set(offset + i, v.test(i));
            }
            offset += v.size();
        };

        (..., append(vals));
#endif
    }

    template <size_t ToW>
    requires (ToW > W)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) explicit constexpr
    operator bitset<ToW>() noexcept {
        bitset<ToW> output;
        output |= *this;
        return output;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) static constexpr size_t size() noexcept {
        return W;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool operator[](
        size_t idx) const noexcept {
        return storage_[idx / chunk_size] & (1zu << (idx % chunk_size));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr bool test(
        size_t idx) const noexcept {
        return *this[idx];
    }

    __DPL_HIDE_FROM_ABI constexpr void set(size_t idx) noexcept {
        auto ptr = storage_ + idx / chunk_size;
        *ptr |= (1zu << (idx % chunk_size));
    }

    __DPL_HIDE_FROM_ABI constexpr void clear(size_t idx) noexcept {
        auto ptr = storage_ + idx / chunk_size;
        *ptr &= ~(1zu << (idx % chunk_size));
    }

    __DPL_HIDE_FROM_ABI constexpr void set(size_t idx, bool val) noexcept {
        if (val) {
            set(idx);
        } else {
            clear(idx);
        }
    }

    __DPL_HIDE_FROM_ABI constexpr bitset& invert() noexcept {
        [this]<size_t I = 0>(this auto self, size_constant<I> = {}) {
            if constexpr (I == chunk_sequence.size() - 1) {
                constexpr auto remainder = W % chunk_size;
                constexpr size_t mask = ~(-1zu << remainder);
                storage_[I] ^= mask;
            } else {
                storage_[I] = ~storage_[I];
            }
        }(chunk_sequence);
    }

    __DPL_HIDE_FROM_ABI constexpr void clear() noexcept {
        [this]<size_t I = 0>(this auto self, size_constant<I> = {}) constexpr {
            if constexpr (I < extent_v<underlying_type>) {
                storage_[I] = 0zu;
                self(size_constant<I + 1>{});
            }
        }();
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
        using underlying = typename bitset<OW>::underlying_type;
        if constexpr (OW <= chunk_size) {
            storage_[0] &= other.value_;
        } else if constexpr (OW < W) {
            [&]<size_t... Is>(index_sequence<Is...>) {
                (..., (__DPL ignore = storage_[Is] &= other.storage_[Is]));
            }(make_index_sequence<extent_v<underlying>>{});
        } else {
            __DPL apply(
                [&](auto... idx) constexpr {
                    (...,
                        (__DPL ignore = storage_[idx] &=
                            other.storage_[idx]));
                },
                extent_sequence);
        }

        return *this;
    }

    template <size_t OW>
    requires (OW <= W)
    __DPL_HIDE_FROM_ABI constexpr bitset& operator|=(
        bitset<OW> const& other) noexcept {
        using underlying = typename bitset<OW>::underlying_type;
        if constexpr (OW <= chunk_size) {
            storage_[0] |= other.value_;
        } else if constexpr (OW < W) {
            [&]<size_t... Is>(index_sequence<Is...>) {
                (..., (__DPL ignore = storage_[Is] |= other.storage_[Is]));
            }(make_index_sequence<extent_v<underlying>>{});
        } else {
            __DPL apply(
                [&](auto... idx) constexpr {
                    (...,
                        (__DPL ignore = storage_[idx] |=
                            other.storage_[idx]));
                },
                extent_sequence);
        }

        return *this;
    }

    template <size_t OW>
    requires (OW <= W)
    __DPL_HIDE_FROM_ABI constexpr bitset& operator^=(
        bitset<OW> const& other) noexcept {
        using underlying = typename bitset<OW>::underlying_type;
        if constexpr (OW <= chunk_size) {
            storage_[0] ^= other.value_;
        } else if constexpr (OW < W) {
            [&]<size_t... Is>(index_sequence<Is...>) {
                (..., (__DPL ignore = storage_[Is] ^= other.storage_[Is]));
            }(make_index_sequence<extent_v<underlying>>{});
        } else {
            __DPL apply(
                [&](auto... idx) constexpr {
                    (...,
                        (__DPL ignore = storage_[idx] ^=
                            other.storage_[idx]));
                },
                extent_sequence);
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
                storage_[i] = storage_[i - word_shift];
            }

            for (auto i = 0zu; i < word_shift; ++i) {
                storage_[i] = 0;
            }
        }

        if (bit_shift != 0) {
            for (auto i = chunks - 1; i > 0; --i) {
                storage_[i] = (storage_[i] << bit_shift) |
                    (storage_[i - 1] >> (chunk_size - bit_shift));
            }

            storage_[0] <<= bit_shift;
        }

        return reinitialize(*this);
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
                storage_[i] = storage_[i + word_shift];
            }

            for (auto i = chunks - word_shift; i < chunks; ++i) {
                storage_[i] = 0;
            }
        }

        if (bit_shift != 0) {
            for (auto i = 0zu; i + 1 < chunks; ++i) {
                storage_[i] = (storage_[i] >> bit_shift) |
                    (storage_[i + 1] << (chunk_size - bit_shift));
            }

            storage_[chunks - 1] >>= bit_shift;
        }

        return *this;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, REINITIALIZES) constexpr bitset&
    reinitialize() noexcept {
        constexpr auto remainder = W % chunk_size;
        if constexpr (remainder > 0) {
            constexpr size_t mask = ~(-1zu << remainder);
            storage_[extent_v<size_t> - 1] &= mask;
        }

        return *this;
    }

private:
    underlying_type storage_;
};

template <same_as<bool>... Bs>
requires (sizeof...(Bs) > 0)
bitset(Bs... vals) -> bitset<sizeof...(Bs)>;

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
        auto const radix = S.radix();
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
