// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.slide;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_algorithm_operation auto slideop, dpp::simd_abi A>
class slide {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = abi_traits<E>::size();
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

    static constexpr auto expected_op(
        auto const& lhs, auto const& rhs, size_t shift) noexcept {
        auto expected = lhs;
        for (auto const i : linear_counter(expected)) {
            if constexpr (slideop == dpp::slide_left) {
                auto const j = i + shift;
                expected[i] =
                    j < expected.size() ? lhs[j] : rhs[j - expected.size()];
            } else {
                auto const j = i - shift;
                expected[i] =
                    j >= expected.size() ? lhs[expected.size() + j] : rhs[j];
            }
        }

        return expected;
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return slide::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;
        test::scalar_generator<size_t> const shift_generator(
            0zu, abi_traits<E>::size());
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);

        for (auto const _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const shift = shift_generator(engine);

            auto const expected = expected_op(lhs, rhs, shift);
            operation_fixture<A>::test(
                test::bitcmp, expected, slideop, lhs, rhs, shift);

            auto const src = data_generator(engine);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(
                slideop, src, vtrue, lhs, rhs, shift);
            operation_fixture<A>::test_masked(
                slideop, src, vfalse, lhs, rhs, shift);
            operation_fixture<A>::test_masked(
                slideop, src, mask, lhs, rhs, shift);
            operation_fixture<A>::test_masked(
                slideop, dpp::zero, mask, lhs, rhs, shift);
            operation_fixture<A>::test(test::bitcmp,
                slideop == dpp::slide_left ? rhs : lhs, slideop, lhs, rhs,
                shift + abi_traits<E>::size());
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const src = data_generator(engine);
            auto const shift = shift_generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(
                    slideop, src, cmask, lhs, rhs, shift);
                operation_fixture<A>::test_masked(
                    slideop, dpp::zero, cmask, lhs, rhs, shift);
            });

            constexpr test::scalar_generator<size_t> cshift_generator(
                0zu, abi_traits<E>::size());
            [&]<size_t I = 0zu, serialized_mt19937 S = {}>(this auto self,
                dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
                if constexpr (I < test_count<E>().size()) {
                    constexpr auto pair = S.generate_with(cshift_generator);
                    constexpr auto cshift = dpp::imm<pair.value>;
                    auto const vmask = mask_generator(engine);
                    auto const expected = expected_op(lhs, rhs, cshift);

                    operation_fixture<A>::test(
                        test::bitcmp, expected, slideop, lhs, rhs, cshift);
                    operation_fixture<A>::test_masked(
                        slideop, src, vtrue, lhs, rhs, cshift);
                    operation_fixture<A>::test_masked(
                        slideop, src, vfalse, lhs, rhs, cshift);
                    operation_fixture<A>::test_masked(
                        slideop, src, vmask, lhs, rhs, cshift);
                    operation_fixture<A>::test_masked(
                        slideop, dpp::zero, vmask, lhs, rhs, cshift);

                    self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
                }
            }
            ();

            // OOB test
            operation_fixture<A>::test(test::bitcmp,
                slideop == dpp::slide_left ? rhs : lhs, slideop, lhs, rhs,
                abi_traits<E>::size);
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using slide_left = slide<dpp::slide_left, A>;
export template <dpp::simd_abi A>
using slide_right = slide<dpp::slide_right, A>;

} // namespace dpl::test