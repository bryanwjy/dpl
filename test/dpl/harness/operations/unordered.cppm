// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.isunordered;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class isunordered {
    template <typename E>
    static constexpr bool expected_op(E lhs, E rhs) noexcept {
        return (lhs != lhs) || (rhs != rhs);
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
                return isunordered::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        static_assert(dpl::floating_point_like<E>);

        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;
        test::scalar_generator<int> const binary_generator(0, 2);
        auto const mfalse = dpp::broadcast<A, E>(false);
        auto const mtrue = dpp::broadcast<A, E>(true);
        for (auto const i : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            operation_fixture<A>::test(mfalse, dpp::cmpunord, lhs, rhs);
        }

        if (test::finite_math_only()) {
            return true;
        }

        for (auto const i : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const rhs = [&]() {
                auto result = lhs;
                for (auto& val : result) {
                    if (binary_generator(engine) == 0) {
                        val = dpp::nan;
                    }
                }

                return result;
            }();

            auto const expected = [&] {
                auto expected = test::make_array<E, A>();
                for (auto const i : linear_counter(expected)) {
                    expected[i] = expected_op(lhs[i], rhs[i])
                        ? dpp::all_bits_v<E>
                        : dpp::zero_v<E>;
                }
                return dpp::cmpneq(dpp::load<A>(expected.data()), dpp::zero);
            }();

            operation_fixture<A>::test(expected, dpp::cmpunord, lhs, rhs);
            operation_fixture<A>::test(expected, dpp::cmpunord, rhs, lhs);
            operation_fixture<A>::test_masked(dpp::cmpunord, mfalse, lhs, rhs);
            operation_fixture<A>::test_masked(dpp::cmpunord, mtrue, rhs, lhs);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(dpp::cmpunord, mask, rhs, lhs);
        }

        operation_fixture<A>::test(
            mfalse, dpp::cmpunord, test::splat<E>(-0.0), 0.0);
        operation_fixture<A>::test(
            mfalse, dpp::cmpunord, 0.0, test::splat<E>(-0.0));
        operation_fixture<A>::test(mfalse, dpp::cmpunord,
            test::splat<E>(dpp::infinity), dpp::infinity);
        operation_fixture<A>::test(mfalse, dpp::cmpunord,
            test::splat<E>(-dpp::infinity), -dpp::infinity);

        {
            auto const finite = data_generator(engine);
            operation_fixture<A>::test(
                mfalse, dpp::cmpunord, test::splat<E>(dpp::infinity), finite);
            operation_fixture<A>::test(
                mfalse, dpp::cmpunord, test::splat<E>(-dpp::infinity), finite);
            operation_fixture<A>::test(
                mfalse, dpp::cmpunord, finite, dpp::infinity);
            operation_fixture<A>::test(
                mfalse, dpp::cmpunord, finite, -dpp::infinity);

            operation_fixture<A>::test(
                mtrue, dpp::cmpunord, finite, dpp::nan_v<E>);
            operation_fixture<A>::test(
                mtrue, dpp::cmpunord, dpp::nan_v<E>, finite);
            operation_fixture<A>::test(mtrue, dpp::cmpunord,
                test::splat(dpp::nan_v<E>), dpp::nan_v<E>);
        }

        return true;
    }
};

} // namespace dpl::test
