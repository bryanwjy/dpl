// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/concepts.h"
#include "dpl/std/utility/bitset/integral_bitset.h"

#if !DPL_MODULES
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/details/bitset.h"
#  include "dpl/std/type_traits/extent.h"
#  include "dpl/std/type_traits/sequence.h"
#endif

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t W>
class bitset : public details::utility::bitset_storage<W> {
    static_assert(W > 0);
    using base_type DPL_NODEBUG = details::utility::bitset_storage<W>;
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

public:
    __DPL_HIDE_FROM_ABI constexpr bitset() noexcept : base_type{} {}

    template <same_as<bool>... Bs>
    requires (sizeof...(Bs) > 0) && (sizeof...(Bs) <= W)
    __DPL_HIDE_FROM_ABI explicit(sizeof...(Bs) != W) constexpr bitset(
        Bs... vals) noexcept
        : base_type{} {
        [&]<size_t... Is>(index_sequence<Is...>) {
            (..., (vals ? set(Is) : clear(Is)));
        }(make_index_sequence<sizeof...(Bs)>{});
    }

    template <size_t... Ws>
    requires (sizeof...(Ws) > 1 && (... + Ws) == W)
    __DPL_HIDE_FROM_ABI constexpr bitset(bitset<Ws> const&... vals) noexcept
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
        size_t shift) const noexcept {
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
            for (auto i = 1zu; i < extent_v<underlying_type>; ++i) {
                this->storage_[i] = 0zu;
            }
        } else if constexpr (integral_bitset_type<bitset<OW>>) {
            this->storage_[0] &= (other.value_ & -1zu);
            this->storage_[1] &= ((other.value_ >> 64) & -1zu);
            for (auto i = 2zu; i < extent_v<underlying_type>; ++i) {
                this->storage_[i] = 0zu;
            }
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
        if constexpr (OW <= chunk_size) {
            this->storage_[0] |= other.value_;
        } else if constexpr (integral_bitset_type<bitset<OW>>) {
            // Only true if underlying of other is uint128
            this->storage_[0] |= (other.value_ & -1zu);
            this->storage_[1] |= ((other.value_ >> 64) & -1zu);
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
        if constexpr (OW <= chunk_size) {
            this->storage_[0] ^= other.value_;
        } else if constexpr (integral_bitset_type<bitset<OW>>) {
            this->storage_[0] ^= (other.value_ & -1zu);
            this->storage_[1] ^= ((other.value_ >> 64) & -1zu);
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

__DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
