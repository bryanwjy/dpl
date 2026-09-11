// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.minmax;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

/**
 * ABI dependent, test drivers should specialize to enable strict tests
 *
 * Only affects floating-point-like elements
 */
export template <dpp::simd_abi A, dpp::simd_element_for<A> E>
inline constexpr bool is_strict_minmax = false;

export template <dpp::simd_primitive_operation auto op, dpp::simd_abi A>
class minmax {
    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept {
        if constexpr (op == dpp::max) {
            return lhs > rhs ? lhs : rhs;
        } else {
            static_assert(op == dpp::min);
            return lhs < rhs ? lhs : rhs;
        }
    }

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        if consteval {
            return linear_counter(4zu);
        } else {
            auto const count = 128zu / dpp::simd_abi_traits<A, E>::size();
            return linear_counter(count < 4zu ? 4zu : count);
        }
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
                return minmax::template run<E>(engine);
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
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);

            auto expected = lhs;
            for (auto const i : linear_counter(expected)) {
                expected[i] = expected_op(lhs[i], rhs[i]);
            }

            test::operation_fixture<A>::test(expected, op, lhs, rhs);

            auto const src = data_generator(engine);
            operation_fixture<A>::test_masked(op, src, vtrue, lhs, rhs);
            operation_fixture<A>::test_masked(op, src, vfalse, lhs, rhs);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(op, src, mask, lhs, rhs);
            operation_fixture<A>::test_masked(op, dpp::zero, mask, lhs, rhs);

            // May not guarantee IEEE behaviour if nan or -0.0 on some
            // architectures
            auto const strict_tests = [&] {
                if constexpr (dpl::floating_point_like<E>) {
                    if constexpr (op == dpp::min) {
                        test::operation_fixture<A>::test(test::bitcmp,
                            test::splat<E>(-0.0), op, test::splat<E>(-0.0),
                            0.0);
                        test::operation_fixture<A>::test(test::bitcmp,
                            test::splat<E>(-0.0), op, test::splat<E>(0.0),
                            -0.0);
                    } else {
                        test::operation_fixture<A>::test(test::bitcmp,
                            test::splat<E>(0.0), op, test::splat<E>(-0.0), 0.0);
                        test::operation_fixture<A>::test(test::bitcmp,
                            test::splat<E>(0.0), op, test::splat<E>(0.0), -0.0);
                    }

                    if (!test::finite_math_only()) {
                        auto const finite = data_generator(engine);
                        test::operation_fixture<A>::test(
                            finite, op, finite, dpp::nan);
                        test::operation_fixture<A>::test(
                            finite, op, dpp::nan, finite);
                    }
                }
            };

            if not consteval {
                if constexpr (!is_strict_minmax<A, E>) {
                    return true;
                }
            }

            strict_tests();
        }

        if constexpr (dpp::fixed_width_abi<A>)
            run_const_mask_test<E>([&](auto cmask) {
                auto const lhs = data_generator(engine);
                auto const rhs = data_generator(engine);
                auto const src = data_generator(engine);

                auto expected = lhs;
                for (auto const i : linear_counter(expected)) {
                    expected[i] = expected_op(lhs[i], rhs[i]);
                }

                operation_fixture<A>::test_masked(op, src, cmask, lhs, rhs);
                operation_fixture<A>::test_masked(
                    op, dpp::zero, cmask, lhs, rhs);
            });

        return true;
    }
};

export template <dpp::simd_abi A>
using minimization = minmax<dpp::min, A>;
export template <dpp::simd_abi A>
using maximization = minmax<dpp::max, A>;

} // namespace dpl::test
