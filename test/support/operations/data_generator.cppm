// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
#include <cstdlib>

export module dpl.test:data_generator;
import dpl;
import :utils.array;
import :utils.unique_array;

namespace dpl::test {

export template <dpl::unsigned_integral T, size_t word_size, size_t state_size,
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

    void update_states() {
        auto i = 0zu;
        for (; i != (state_size - shift_size); ++i)
            update_state(i);
        for (auto j = 0zu; i != state_size; ++i, ++j)
            update_state(i, j);
    }

    void update_state(size_t i, size_t k) {
        size_t const j = (i + 1) % state_size;
        auto const mask = static_cast<result_type>(mask_bits == digits
                ? ~static_cast<result_type>(0)
                : (static_cast<result_type>(1) << mask_bits) -
                    static_cast<result_type>(1));

        auto const tmp = (state_[i] & ~mask) | (state_[j] & mask);
        state_[i] = state_[k] ^ rshift<1>(tmp) ^ (xor_mask * (tmp & 1));
    }

    void update_state(size_t i) {
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
        auto val = state_[idx_];
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

export using mt19937 = mt_engine<dpl::uint64, 64, 312, 156, 31,
    0xb5026f5aa96619e9ull, 29, 0x5555555555555555ull, 17, 0x71d67fffeda60000ull,
    37, 0xfff7eee000000000ull, 43, 6364136223846793005ull>;

namespace dpp = dpl::datapar;
export class data_generator {

public:
    template <typename E>
    static constexpr E generate(E min, E max, mt19937& rng) noexcept {
        auto const delta = max - min;
        if constexpr (dpl::integral<E>) {
            return delta == 0
                ? min
                : static_cast<E>(rng() % static_cast<dpl::uint64>(delta)) + min;
        } else {
            using bitset_t = dpl::bitset<floating_point_traits<E>::width>;
            max = max == dpp::infinity_v<E> ? dpp::max_value_v<E> : max;
            min = min == -dpp::infinity_v<E> ? dpp::min_value_v<E> : min;
            auto const bits = dpl::bit_cast<bitset_t>(delta);
            auto const exp = dpl::to_underlying(
                (bits & floating_point_traits<E>::exponent_mask) >>
                floating_point_traits<E>::digits);
            auto const fr = dpl::to_underlying(
                bits & floating_point_traits<E>::mantissa_mask);
            constexpr auto zero = dpl::to_underlying(bitset_t());
            auto const rand_exp = bitset_t(generate(zero, exp, rng))
                << floating_point_traits<E>::digits;
            auto const rand_fr = bitset_t(generate(zero, fr, rng));
            auto const sign_condition = dpl::popcount(rng()) >
                (sizeof(decltype(rng())) * dpl::char_bit_v / 2);
            auto const sign = min < 0 && (max < 0 || sign_condition)
                ? dpl::bit_cast<bitset_t>(static_cast<E>(-0.0))
                : bitset_t();
            return dpl::bit_cast<E>(rand_exp | rand_fr | sign) + min;
        }
    }

    template <typename E>
    static constexpr E generate(mt19937& rng) noexcept {
        if constexpr (dpl::integral<E>) {
            return static_cast<E>(rng());
        } else {
            return generate<E>(-0.0, dpp::max_value, rng);
        }
    }

    template <dpp::simd_abi A, dpp::simd_element_for<A> E>
    static constexpr auto generate_array(E min, E max, mt19937& rng) noexcept {
        using abi_traits = dpp::simd_abi_traits<A, E>;
        if constexpr (dpp::fixed_width_abi<A>) {
            using array_t = array<E, abi_traits::size>;
            return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
                return array_t{(dpl::ignore = Is, generate(min, max, rng))...};
            }(dpl::make_index_sequence<abi_traits::size>{});
        } else {
            unique_array<E> data(abi_traits::size());
            for (auto& val : data) {
                val = generate(min, max, rng);
            }
            return data;
        }
    }

    template <dpp::simd_abi A, dpp::simd_element_for<A> E>
    static constexpr auto generate_array(mt19937& rng) noexcept {
        using abi_traits = dpp::simd_abi_traits<A, E>;
        if constexpr (dpp::fixed_width_abi<A>) {
            using array_t = array<E, abi_traits::size>;
            return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
                return array_t{(dpl::ignore = Is, generate<E>(rng))...};
            }(dpl::make_index_sequence<abi_traits::size>{});
        } else {
            unique_array<E> data(abi_traits::size());
            for (auto& val : data) {
                val = generate<E>(rng);
            }
            return data;
        }
    }
};

} // namespace dpl::test
