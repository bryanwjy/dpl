// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.signop;
export import dpl.test.support;

import dpl;

export namespace dpl::test {
namespace dpp = dpl::datapar;

template <dpp::simd_primitive_operation auto op, dpp::simd_abi A>
class signop {

    template <typename E>
    static constexpr E expected_op(E arg) noexcept
    requires (op == dpp::abs)
    {
        if constexpr (dpl::unsigned_integral<E>) {
            return arg;
        } else if constexpr (dpl::integral<E>) {
            using U = dpp::unsigned_representation_t<E>;
            return arg < E(0) ? static_cast<E>(-static_cast<U>(arg)) : arg;
        } else {
            constexpr auto signbit = dpl::floating_point_traits<E>::signbit;
            return dpl::bit_cast<E>(dpl::to_bit_representation(arg) & ~signbit);
        }
    }

    template <typename E>
    static constexpr E expected_op(E arg) noexcept
    requires (op == dpp::negate)
    {
        if constexpr (dpl::integral<E>) {
            return -arg;
        } else {
            constexpr auto signbit = dpl::floating_point_traits<E>::signbit;
            return dpl::bit_cast<E>(dpl::to_bit_representation(arg) ^ signbit);
        }
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
                return signop::template run<E>(engine);
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
            auto const arg = data_generator(engine);
            auto const src = data_generator(engine);
            auto expected = arg;
            for (auto const i : linear_counter(expected)) {
                expected[i] = expected_op(arg[i]);
            }

            test::operation_fixture<A>::test(test::bitcmp, expected, op, arg);
            test::operation_fixture<A>::test_masked(op, src, vtrue, arg);
            test::operation_fixture<A>::test_masked(op, src, vfalse, arg);
            auto const mask = mask_generator(engine);
            test::operation_fixture<A>::test_masked(op, src, mask, arg);
            test::operation_fixture<A>::test_masked(op, dpp::zero, mask, arg);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const arg = data_generator(engine);
            auto const src = data_generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                test::operation_fixture<A>::test_masked(op, src, cmask, arg);
                test::operation_fixture<A>::test_masked(
                    op, dpp::zero, cmask, arg);
            });
        }

        // INT_MIN: abs(INT_MIN) == INT_MIN on 2's complement hardware (wraps).
        // Float special cases (-0.0, ±inf, NaN) compared via bit_cast since
        // NaN != NaN under IEEE 754.
        if constexpr (dpl::signed_integral<E>) {
            if not consteval {
                // Implementation-defined
                auto const min = dpl::integral_traits<E>::min_value;
                auto const vmin = dpp::broadcast<A, E>(min);
                test::operation_fixture<A>::test(vmin, op, vmin);
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            for (auto const arg :
                test::array{dpp::msb_v<E>, -dpp::infinity_v<E>,
                    dpp::infinity_v<E>, -dpp::nan_v<E>, dpp::nan_v<E>}) {

                auto const expected = dpp::broadcast<A, E>(expected_op(arg));
                auto const varg = dpp::broadcast<A, E>(arg);
                test::operation_fixture<A>::test(
                    test::bitcmp, expected, op, varg);
            }
        }

        return true;
    }
};

template <dpp::simd_abi A>
using abs = signop<dpp::abs, A>;
template <dpp::simd_abi A>
using negate = signop<dpp::negate, A>;

} // namespace dpl::test
