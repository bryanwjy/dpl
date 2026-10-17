// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.math.copysign;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class copysign {
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept {
        if constexpr (dpl::unsigned_integral<E>) {
            return lhs;
        } else if constexpr (dpl::signed_integral<E>) {
            auto const lb = dpl::to_bit_representation(lhs);
            auto const rb = dpl::to_bit_representation(rhs);
            constexpr auto signbit = dpl::integral_traits<E>::signbit;
            return dpl::bit_cast<E>((lb & ~signbit) | (rb & signbit));
        } else {
            auto const lb = dpl::to_bit_representation(lhs);
            auto const rb = dpl::to_bit_representation(rhs);
            constexpr auto signbit = dpl::floating_point_traits<E>::signbit;
            return dpl::bit_cast<E>((lb & ~signbit) | (rb & signbit));
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
                return copysign::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        constexpr array_generator<A, E> generator;
        constexpr mask_generator<A, E> mask_generator;
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);

        for (auto const _ : test_count<E>()) {
            auto const lhs = generator(engine);
            auto const rhs = generator(engine);
            auto const src = generator(engine);
            auto expected = lhs;
            for (auto const i : linear_counter(expected)) {
                expected[i] = expected_op(lhs[i], rhs[i]);
            }

            operation_fixture<A>::test(
                test::bitcmp, expected, dpp::copysign, lhs, rhs);
            operation_fixture<A>::test_masked(
                dpp::copysign, src, vtrue, lhs, rhs);
            operation_fixture<A>::test_masked(
                dpp::copysign, src, vfalse, lhs, rhs);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(
                dpp::copysign, src, mask, lhs, rhs);
            operation_fixture<A>::test_masked(
                dpp::copysign, dpp::zero, mask, lhs, rhs);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const lhs = generator(engine);
            auto const rhs = generator(engine);
            auto const src = generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(
                    dpp::copysign, src, cmask, lhs, rhs);
                operation_fixture<A>::test_masked(
                    dpp::copysign, dpp::zero, cmask, lhs, rhs);
            });
        }

        return true;
    }
};

} // namespace dpl::test
