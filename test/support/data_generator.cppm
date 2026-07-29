// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.support:data_generator;
import dpl;
import :array;
import :dynamic_array;
import :span;
import :bitset_helpers;

namespace dpl::test {
namespace dpp = dpl::datapar;
}

export namespace dpl::test {
inline namespace support {

template <dpl::unsigned_integral T, size_t word_size, size_t state_size,
    size_t shift_size, size_t mask_bits, T xor_mask, size_t u, T d, size_t s,
    T b, size_t t, T c, size_t l, T multiplier>
class mt_engine {
public:
    using result_type = T;

private:
    result_type state_[state_size]{};
    size_t idx_ = 0;
    static constexpr result_type digits =
        dpl::integral_traits<result_type>::digits;
    static constexpr result_type lower_bound = 0;
    static constexpr result_type upper_bound = static_cast<result_type>(
        word_size == digits ? ~static_cast<result_type>(0)
                            : (static_cast<result_type>(1) << word_size) -
                static_cast<result_type>(1));

    static_assert(0 < shift_size, "invalid parameters");
    static_assert(shift_size <= state_size, "invalid parameters");
    static_assert(word_size <= digits, "invalid parameters");
    static_assert(2 <= word_size, "invalid parameters");
    static_assert(mask_bits <= word_size, "invalid parameters");
    static_assert(u <= word_size, "invalid parameters");
    static_assert(s <= word_size, "invalid parameters");
    static_assert(t <= word_size, "invalid parameters");
    static_assert(l <= word_size, "invalid parameters");
    static_assert(lower_bound < upper_bound, "invalid parameters");
    static_assert(xor_mask <= upper_bound, "invalid parameters");
    static_assert(b <= upper_bound, "invalid parameters");
    static_assert(c <= upper_bound, "invalid parameters");
    static_assert(d <= upper_bound, "invalid parameters");
    static_assert(multiplier <= upper_bound, "invalid parameters");

    template <size_t N>
    static constexpr result_type lshift(result_type val) {
        if constexpr (N < word_size) {
            return (val << N) & upper_bound;
        } else {
            return 0;
        }
    }

    template <size_t N>
    static constexpr result_type rshift(result_type val) {
        if constexpr (N < digits) {
            return val >> N;
        } else {
            return 0;
        }
    }

    constexpr void update_states() {
        auto i = 0zu;
        for (; i != (state_size - shift_size); ++i)
            update_state(i);
        for (auto j = 0zu; i != state_size; ++i, ++j)
            update_state(i, j);
    }

    constexpr void update_state(size_t i, size_t k) {
        size_t const j = (i + 1) % state_size;
        auto const mask = static_cast<result_type>(mask_bits == digits
                ? ~static_cast<result_type>(0)
                : (static_cast<result_type>(1) << mask_bits) -
                    static_cast<result_type>(1));

        auto const tmp = (state_[i] & ~mask) | (state_[j] & mask);
        state_[i] = state_[k] ^ rshift<1>(tmp) ^ (xor_mask * (tmp & 1));
    }

    constexpr void update_state(size_t i) {
        update_state(i, (i + shift_size) % state_size);
    }

public:
    [[nodiscard]] static constexpr result_type min() noexcept {
        return lower_bound;
    }
    [[nodiscard]] static constexpr result_type max() noexcept {
        return upper_bound;
    }
    static constexpr result_type default_seed = 919393u;

    constexpr mt_engine(result_type seed = default_seed) noexcept {
        this->seed(seed);
    }

    constexpr void seed(result_type seed = default_seed) noexcept {
        state_[0] = seed & upper_bound;
        for (auto i = 1zu; i < state_size; ++i) {
            state_[i] =
                (multiplier *
                        (state_[i - 1] ^ rshift<word_size - 2>(state_[i - 1])) +
                    i) &
                upper_bound;
        }
        idx_ = 0;
    }
    [[nodiscard]] constexpr result_type operator()() noexcept {
        auto const val = state_[idx_];
        if (++idx_ == state_size) {
            update_states();
            idx_ = 0;
        }
        result_type lhs = val ^ (rshift<u>(val) & d);
        lhs ^= lshift<s>(lhs) & b;
        lhs ^= lshift<t>(lhs) & c;
        return lhs ^ rshift<l>(lhs);
    }

    constexpr void discard(size_t n) {
        for (; n > 0zu; --n) {
            dpl::ignore = operator()();
        }
    }
};

using mt19937 = mt_engine<dpl::uint64, 64, 312, 156, 31, 0xb5026f5aa96619e9ull,
    29, 0x5555555555555555ull, 17, 0x71d67fffeda60000ull, 37,
    0xfff7eee000000000ull, 43, 6364136223846793005ull>;

struct half_range_t {
    explicit constexpr half_range_t() noexcept = default;
};

inline constexpr half_range_t half_range{};

template <typename T>
concept rng_like = requires(T& rng) {
    typename T::result_type;
    requires dpl::unsigned_integral<typename T::result_type>;
    { rng() } noexcept -> dpl::same_as<typename T::result_type>;
};

template <typename E>
class scalar_generator {
    static_assert(dpl::integral<E> || dpl::floating_point_like<E>);

public:
    static constexpr E minimum = []() {
        if constexpr (dpl::integral<E>) {
            return dpl::integral_traits<E>::min_value;
        } else {
            return -dpp::max_value_v<E>;
        }
    }();

    static constexpr E maximum = []() {
        if constexpr (dpl::integral<E>) {
            return dpl::integral_traits<E>::max_value;
        } else {
            return dpp::max_value_v<E>;
        }
    }();

private:
    static constexpr E subtract(E lhs, E rhs) noexcept
    requires dpl::integral<E>
    {
        if constexpr (dpl::signed_integral<E>) {
            constexpr auto shift = dpl::type_bit_v<E> - 1;
            auto const inter =
                dpl::to_signed(static_cast<dpl::make_unsigned_t<E>>(
                    dpl::to_unsigned(lhs) - dpl::to_unsigned(rhs)));

            auto const lsign = lhs >> shift;
            auto const rsign = rhs >> shift;
            auto const isign = inter >> shift;
            auto const overflowed =
                static_cast<bool>((lsign ^ rsign) & (lsign ^ isign));
            return overflowed ? maximum : inter;
        } else {
            return lhs - rhs;
        }
    }

public:
    constexpr scalar_generator() noexcept : min_(minimum), max_(maximum) {}

    constexpr scalar_generator(E min, E max) noexcept : min_(min), max_(max) {}

    explicit constexpr scalar_generator(half_range_t) noexcept
        : min_(minimum / 2)
        , max_(maximum / 2) {}

    template <rng_like Rng>
    constexpr E operator()(Rng& rng) const noexcept {
        using rtype = typename Rng::result_type;
        if constexpr (dpl::integral<E>) {
            auto const delta = subtract(max_, min_);
            return delta == 0
                ? min_
                : static_cast<E>(rng() % static_cast<rtype>(delta)) + min_;
        } else {
            using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
            using uint_t = dpp::unsigned_representation_t<E>;
            auto const max = max_ == dpp::infinity_v<E> ? dpp::max_value : max_;
            auto const min =
                min_ == -dpp::infinity_v<E> ? dpp::min_value : min_;
            if (!(min_ <= max_)) [[unlikely]] {
                return dpp::nan;
            }

            auto const signed_min = dpl::bit_cast<bitset_t>(min_) &
                floating_point_traits<E>::signbit;
            auto const signed_max = dpl::bit_cast<bitset_t>(max) &
                floating_point_traits<E>::signbit;

            constexpr auto exp_bits = floating_point_traits<E>::exponent_mask;
            constexpr auto fr_bits = floating_point_traits<E>::mantissa_mask;
            constexpr auto exp_rzero = dpl::countr_zero(exp_bits);
            constexpr auto bias = floating_point_traits<E>::exponent_bias;

            if (signed_max == signed_min) {
                auto const bits = dpl::bit_cast<bitset_t>(max - min);
                auto const exp =
                    dpl::to_underlying((bits & exp_bits) >> exp_rzero);
                auto const fr = dpl::to_underlying(bits & fr_bits);

                auto const rand_exp =
                    bitset_t(scalar_generator<uint_t>(1, exp)(rng))
                    << exp_rzero;
                auto const rand_fr =
                    bitset_t(scalar_generator<uint_t>(0, fr)(rng));
                return dpl::bit_cast<E>(rand_exp | rand_fr) + min;
            } else {

                auto const is_signed = [&]() {
                    auto const split = dpl::bit_cast<uint_t>(-min);
                    auto const domain = dpl::bit_cast<uint_t>(max) + split - 1;
                    auto gen = scalar_generator<uint_t>(0, domain);
                    return gen(rng) < split;
                }();
                auto const bits = is_signed ? dpl::bit_cast<bitset_t>(-min)
                                            : dpl::bit_cast<bitset_t>(max);
                auto const signbit = is_signed
                    ? dpl::bit_cast<bitset_t>(static_cast<E>(-0.0))
                    : bitset_t();

                auto const exp =
                    dpl::to_underlying((bits & exp_bits) >> exp_rzero);
                auto const fr = dpl::to_underlying(bits & fr_bits);
                auto const rand_exp =
                    bitset_t(scalar_generator<uint_t>(1, exp)(rng))
                    << exp_rzero;
                auto const rand_fr =
                    bitset_t(scalar_generator<uint_t>(0, fr)(rng));
                return dpl::bit_cast<E>(signbit | rand_exp | rand_fr);
            }
        }
    }

private:
    E min_;
    E max_;
};

template <dpp::simd_abi A, dpp::simd_element_for<A> E>
class array_generator : private scalar_generator<E> {
    using abi_traits = dpp::simd_abi_traits<A, E>;
    using array_t = array<E, abi_traits::size>;
    using base_type = scalar_generator<E>;

public:
    using base_type::base_type;

    template <rng_like Rng>
    constexpr array_t operator()(Rng& rng) const
        noexcept(dpp::fixed_width_abi<A>) {
        if constexpr (dpp::fixed_width_abi<A>) {
            return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
                return array_t{
                    (dpl::ignore = Is, base_type::operator()(rng))...};
            }(dpl::make_index_sequence<abi_traits::size>{});
        } else {
            dynamic_array<E> data(abi_traits::size());
            for (auto& val : data) {
                val = base_type::operator()(rng);
            }
            return data;
        }
    }

    template <rng_like Rng>
    constexpr E scalar(Rng& rng) const noexcept {
        return base_type::operator()(rng);
    }
};

template <size_t N>
class bit_generator;

template <size_t N>
requires dpl::integral_bitset_type<dpl::bitset<N>>
class bit_generator<N> :
    private scalar_generator<typename dpl::bitset<N>::underlying_type> {
    using result_type = dpl::bitset<N>;
    using underlying_type = typename result_type::underlying_type;
    using base_type = scalar_generator<underlying_type>;

public:
    constexpr bit_generator() noexcept
        : base_type(static_cast<underlying_type>(0),
              static_cast<underlying_type>(
                  static_cast<underlying_type>(1) << N)) {}

    template <rng_like Rng>
    constexpr result_type operator()(Rng& rng) const noexcept {
        return result_type{base_type::operator()(rng)};
    }
};

template <size_t N>
class bit_generator : private scalar_generator<size_t> {
    using result_type = dpl::bitset<N>;
    using base_type = scalar_generator<size_t>;

public:
    constexpr bit_generator() noexcept : scalar_generator() {}

    template <rng_like Rng>
    constexpr result_type operator()(Rng& rng) const noexcept {
        constexpr auto remainder = N % dpl::type_bit_v<size_t>;
        constexpr auto count = N / dpl::type_bit_v<size_t> + remainder > 0;
        return [&]<size_t... Is>(dpl::index_sequence<Is..., count - 1>) {
            return result_type{
                bit_representation_t<decltype(Is)>(
                    base_type::operator()(rng))...,
                bit_representation_t<size_t>(
                    base_type::operator()(rng) % remainder),
            };
        }(dpl::make_index_sequence<count>{});
    }
};
} // namespace support
} // namespace dpl::test
