// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.clamp;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class clamp {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr E expected_op(E val, E min, E max) noexcept {
        val = val < min ? min : val;
        val = val > max ? max : val;
        return val;
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

                func(cmask);
                self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
            }
        }
        ();
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return clamp::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;

        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);
        for (auto const _ : test_count<E>()) {
            auto minval = data_generator(engine);
            auto maxval = data_generator(engine);
            for (auto const i : linear_counter(maxval)) {
                if (minval[i] > maxval[i]) {
                    dpl::ranges::swap(minval[i], maxval[i]);
                }
            }

            auto const val = data_generator(engine);
            auto const src = data_generator(engine);

            auto expected = val;
            for (auto const i : linear_counter(expected)) {
                expected[i] = expected_op(val[i], minval[i], maxval[i]);
            }

            operation_fixture<A>::test(
                test::bitcmp, expected, dpp::clamp, val, minval, maxval);
            operation_fixture<A>::test_masked(
                dpp::clamp, src, vtrue, val, minval, maxval);
            operation_fixture<A>::test_masked(
                dpp::clamp, src, vfalse, val, minval, maxval);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(
                dpp::clamp, src, mask, val, minval, maxval);
            operation_fixture<A>::test_masked(
                dpp::clamp, dpp::zero, mask, val, minval, maxval);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto minval = data_generator(engine);
            auto maxval = data_generator(engine);
            for (auto const i : linear_counter(maxval)) {
                if (minval[i] > maxval[i]) {
                    dpl::ranges::swap(minval[i], maxval[i]);
                }
            }

            auto const val = data_generator(engine);
            auto const src = data_generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(
                    dpp::clamp, src, cmask, val, minval, maxval);
                operation_fixture<A>::test_masked(
                    dpp::clamp, dpp::zero, cmask, val, minval, maxval);
            });
        }

        return true;
    }
};
} // namespace dpl::test
