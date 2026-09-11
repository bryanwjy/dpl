// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.reduction;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_algorithm_operation auto rop, dpp::simd_abi A,
    dpp::simd_element_for<A> E>
struct reduction_data_generator : test::array_generator<A, E> {
    using test::array_generator<A, E>::array_generator;
    using test::array_generator<A, E>::operator();
};

export template <dpp::simd_algorithm_operation auto rop, dpp::simd_abi A,
    dpp::simd_element_for<A> E>
requires (rop == dpp::hsum && dpl::integral<E>)
struct reduction_data_generator<rop, A, E> {
    template <typename Rng>
    constexpr auto operator()(Rng& engine) const noexcept {
        return gen(engine);
    }

private:
    static constexpr E range =
        dpl::integral_traits<E>::max_value / dpp::simd_abi_traits<A, E>::size;
    test::array_generator<A, E> gen =
        test::array_generator<A, E>(dpl::is_signed_v<E> ? -range : 0, range);
};

export template <dpp::simd_algorithm_operation auto rop, dpp::simd_abi A,
    dpp::simd_element_for<A> E>
requires (rop == dpp::hsum && dpl::floating_point_like<E>)
struct reduction_data_generator<rop, A, E> {

public:
    template <typename Rng>
    constexpr auto operator()(Rng& engine) const noexcept {
        auto const is_neg = sign_gen(engine);
        auto const frac = frac_gen(engine);
        using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
        auto const exp = dpl::bit_cast<uint_t>(
            static_cast<bitset_t>(exp_gen(engine))
            << dpl::countr_zero(dpl::floating_point_traits<E>::exponent_mask));

        constexpr auto shift = dpl::type_bit_v<E> - 1;
        auto result = result_gen(engine);
        for (auto i = 0zu; i < frac.size(); ++i) {
            auto const val =
                static_cast<uint_t>((is_neg << shift) | frac[i] | exp);
            result[i] = __DPL bit_cast<E>(val);
        }

        return result;
    }

private:
    using uint_t = dpp::unsigned_representation_t<E>;
    static constexpr auto nexp =
        dpl::popcount(dpl::floating_point_traits<E>::exponent_mask);
    static constexpr uint_t frac_range =
        dpl::to_underlying(dpl::floating_point_traits<E>::mantissa_mask) /
        dpp::simd_abi_traits<A, E>::size;

    test::bit_generator<nexp> exp_gen;
    test::scalar_generator<uint_t> sign_gen =
        test::scalar_generator<uint_t>(0, 2);
    test::array_generator<A, uint_t> frac_gen =
        test::array_generator<A, uint_t>(0, frac_range);
    test::array_generator<A, E> result_gen;
};

export template <dpp::simd_abi A, dpp::simd_algorithm_operation auto rop>
class reduction {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using array_t = decltype(test::make_array<E, A>());
    template <typename E>
    using mask_t = decltype(dpp::broadcast<A, E>(true));

    template <dpp::simd_element_for<A> E>
    static constexpr E expected_op(array_t<E> const& val) noexcept {
        auto result = [&] {
            if constexpr (dpl::floating_point_like<E> && rop == dpp::hsum) {
                // TODO deal with MSVC
                return static_cast<long double>(val[0]);
            } else {
                return val[0];
            }
        }();
        for (auto i = 1zu; i < val.size(); ++i) {
            if constexpr (rop == dpp::hsum) {
                result += val[i];
            } else if constexpr (rop == dpp::hmax) {
                result = result < val[i] ? val[i] : result;
            } else {
                result = result > val[i] ? val[i] : result;
            }
        }

        return static_cast<E>(result);
    }

    template <dpp::simd_element_for<A> E>
    static constexpr E expected_op(
        array_t<E> const& val, mask_t<E> mask) noexcept {
        auto const first = dpp::countr_zero(mask);
        auto result = [&] {
            if constexpr (dpl::floating_point_like<E> && rop == dpp::hsum) {
                // TODO deal with MSVC
                return static_cast<long double>(val[first]);
            } else {
                return val[first];
            }
        }();
        auto const amask = to_mask_array<E, A>(mask);
        for (auto i = first + 1; i < val.size(); ++i) {
            if (amask[i] != 0) {
                if constexpr (rop == dpp::hsum) {
                    result += val[i];
                } else if constexpr (rop == dpp::hmax) {
                    result = result < val[i] ? val[i] : result;
                } else {
                    result = result > val[i] ? val[i] : result;
                }
            }
        }

        return result;
    }

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = dpp::simd_abi_traits<A, E>::size();
        auto const limit = []() {
            if consteval {
                return 4zu;
            } else {
                return 128zu;
            }
        }();

        auto max = lanes >= dpl::type_bit_v<size_t>
            ? limit
            : static_cast<size_t>((1zu << lanes) - 1);
        return linear_counter(max < limit ? max : limit);
    }

    template <typename E>
    static constexpr void run_const_mask_test(auto func) noexcept {
        [&]<size_t I = 0, serialized_mt19937 S = {}>(this auto self,
            dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
            if constexpr (I < test_count<E>().size()) {
                constexpr auto lanes = dpp::simd_abi_traits<A, E>::size();
                constexpr bit_generator<lanes> bitgen;
                constexpr auto pair = S.generate_with(bitgen);
                constexpr auto cmask = dpp::deduce_const_mask_v<pair.value>;

                if constexpr (dpp::none_of(cmask)) {
                    self(dpp::imm<I>, mt19937_type<pair.state>{});
                } else {
                    func(cmask);
                    self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
                }
            }
        }
        ();
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return reduction::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        constexpr auto width = abi_traits<E>::size();
        reduction_data_generator<rop, A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;

        for (auto const _ : test_count<E>()) {
            auto const val = data_generator(engine);
            auto const mask = [&] {
                auto mask = mask_generator(engine);
                while (dpp::none_of(mask)) {
                    mask = mask_generator(engine);
                }
                return mask;
            }();

            {

                auto const expected = expected_op<E>(val);
                test::operation_fixture<A>::test(
                    precision_cmp<2>, expected, rop, val);
            }
            {
                auto const expected = expected_op<E>(val, mask);
                test::operation_fixture<A>::test(
                    precision_cmp<2>, expected, rop, val, mask);
            }
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const val = data_generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                auto const expected = expected_op<E>(val, cmask);
                test::operation_fixture<A>::test(
                    precision_cmp<2>, expected, rop, val, cmask);
            });
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using hsum = reduction<A, dpp::hsum>;
export template <dpp::simd_abi A>
using hmin = reduction<A, dpp::hmin>;
export template <dpp::simd_abi A>
using hmax = reduction<A, dpp::hmax>;

export template <dpp::simd_abi A>
class reduce {

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = dpp::simd_abi_traits<A, E>::size();
        auto const limit = []() {
            if consteval {
                return 4zu;
            } else {
                return 128zu;
            }
        }();

        auto max = lanes >= dpl::type_bit_v<size_t>
            ? limit
            : static_cast<size_t>((1zu << lanes) - 1);
        return linear_counter(max < limit ? max : limit);
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return reduce::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        reduction_data_generator<dpp::hmax, A, E> const max_generator;
        reduction_data_generator<dpp::hmin, A, E> const min_generator;

        for (auto const _ : test_count<E>()) {
            {
                auto const val = dpp::load<A>(max_generator(engine).data());
                auto const expected = dpp::hmax(val);
                test::operation_fixture<A>::test(
                    precision_cmp<2>, expected, dpp::reduce, val, dpp::max);
            }
            {
                auto const val = dpp::load<A>(min_generator(engine).data());
                auto const expected = dpp::hmin(val);
                test::operation_fixture<A>::test(
                    precision_cmp<2>, expected, dpp::reduce, val, dpp::min);
            }
        }

        return true;
    }
};

} // namespace dpl::test
