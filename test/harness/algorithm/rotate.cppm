// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.rotate;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_algorithm_operation auto rotateop, dpp::simd_abi A>
class lane_rotate {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using array_t = decltype(test::make_array<E, A>());

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

    template <typename E>
    static constexpr array_t<E> expected_op(
        array_t<E> const& lhs, size_t shift) {
        auto expected = lhs;
        for (auto const i : linear_counter(expected)) {
            if constexpr (rotateop == dpp::rotate_left) {
                auto const j = i + shift;
                expected[i] =
                    j < expected.size() ? lhs[j] : lhs[j - expected.size()];
            } else {
                auto const j = i - shift;
                expected[i] =
                    j >= expected.size() ? lhs[expected.size() + j] : lhs[j];
            }
        }
        return expected;
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return lane_rotate::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;
        test::scalar_generator<size_t> const rot_generator(
            0zu, abi_traits<E>::size());
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);

        for (auto const _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const rhs = rot_generator(engine);
            auto const src = data_generator(engine);
            auto const expected = expected_op<E>(lhs, rhs);

            operation_fixture<A>::test(
                test::bitcmp, expected, rotateop, lhs, rhs);
            operation_fixture<A>::test(test::bitcmp, expected, rotateop, lhs,
                rhs + abi_traits<E>::size());
            operation_fixture<A>::test_masked(rotateop, src, vtrue, lhs, rhs);
            operation_fixture<A>::test_masked(rotateop, src, vfalse, lhs, rhs);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(rotateop, src, mask, lhs, rhs);
            operation_fixture<A>::test_masked(
                rotateop, dpp::zero, mask, lhs, rhs);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const lhs = data_generator(engine);
            auto const rhs = rot_generator(engine);
            auto const src = data_generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(
                    rotateop, src, cmask, lhs, rhs);
                operation_fixture<A>::test_masked(
                    rotateop, dpp::zero, cmask, lhs, rhs);
            });

            [&]<size_t I = 0zu, serialized_mt19937 S = {}>(this auto self,
                dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
                if constexpr (I < test_count<E>().size()) {
                    constexpr test::scalar_generator<size_t> crot_generator(
                        0zu, abi_traits<E>::size());

                    constexpr auto pair = S.generate_with(crot_generator);
                    constexpr auto crhs = dpp::imm<pair.value>;
                    auto const vmask = mask_generator(engine);
                    auto const expected = expected_op<E>(lhs, crhs);

                    operation_fixture<A>::test(
                        test::bitcmp, expected, rotateop, lhs, crhs);
                    operation_fixture<A>::test_masked(
                        rotateop, src, vtrue, lhs, crhs);
                    operation_fixture<A>::test_masked(
                        rotateop, src, vfalse, lhs, crhs);
                    operation_fixture<A>::test_masked(
                        rotateop, src, vmask, lhs, crhs);
                    operation_fixture<A>::test_masked(
                        rotateop, dpp::zero, vmask, lhs, crhs);

                    self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
                }
            }
            ();
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using rotate_left = lane_rotate<dpp::rotate_left, A>;
export template <dpp::simd_abi A>
using rotate_right = lane_rotate<dpp::rotate_right, A>;

} // namespace dpl::test
