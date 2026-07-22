// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test:fused_multiply;
import :support.ternary_transform;
import :support.span;
import :support.comparison;
import :support.bitset_helpers;
import :support.data_generator;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class fused_multiply {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;

public:
    template <dpp::simd_primitive_operation auto fmop, rng_like Rng,
        dpl::floating_point_like... Es>
    static constexpr bool run_all(
        dpl::type_pack<Es...> pack, auto expected_op, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                static_assert(
                    dpp::is_simd_invocable<vec_t<E>, vec_t<E>, vec_t<E>>(fmop));
                return fused_multiply::template run<E, fmop>(
                    expected_op, engine);
            },
            pack);
    }

    template <dpl::floating_point_like E,
        dpp::simd_primitive_operation auto fmop, rng_like Rng>
    static constexpr bool run(auto expected_op, Rng& engine) {

        constexpr auto max = []() {
            constexpr auto bias = dpl::floating_point_traits<E>::exponent_bias;
            constexpr auto exp = dpl::floating_point_traits<E>::exponent_mask;
            constexpr auto shift = dpl::countr_zero(exp);
            // Use quad root instead of square root
            constexpr auto qrt =
                dpl::bitset<dpl::type_bit_v<E>>(
                    ((dpl::to_underlying(exp >> shift) - bias) >> 2) + bias - 1)
                << shift;
            return dpl::bit_cast<E>(qrt);
        }();
        constexpr auto min = -max;

        dpl::test::array_generator<abi_t, E> const data_generator(min, max);
        dpl::test::scalar_generator<E> const src_generator(
            max * max * max, dpp::max_value_v<E>);

        auto const lhs = data_generator(engine);
        auto const mid = data_generator(engine);
        auto const rhs = data_generator(engine);
        auto const src = src_generator(engine);

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op(lhs[i], mid[i], rhs[i]);
        }

        dpl::test::ternary_transform<abi_t>::template test<E>(
            lhs, mid, rhs, fmop, expected);
        dpl::test::ternary_transform<abi_t>::template test_masked<E>(
            lhs, mid, rhs, fmop, src);

        dpl::test::ternary_transform<abi_t>::template test<E>(1, 1, 0, fmop, 1);
        dpl::test::ternary_transform<abi_t>::template test<E>(1, 1, 1, fmop, 2);
        {
            auto const a = src_generator(engine), b = src_generator(engine);
            dpl::test::ternary_transform<abi_t>::template test<E>(
                a, b, 0, fmop, a * b);
        }
        if constexpr (dpp::is_simd_canonical_invocable<vec_t<E>, vec_t<E>,
                          vec_t<E>>(dpp::fmadd))
            if not consteval {
                // catostrophic cancellation test
                constexpr auto small_exp =
                    floating_point_traits<E>::exponent_bias -
                    (floating_point_traits<E>::digits - 1);
                using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
                constexpr auto small = bitset_t(small_exp)
                    << dpl::countr_zero(
                           floating_point_traits<E>::exponent_mask);

                if constexpr (fmop == dpp::fmadd) {
                    dpl::test::ternary_transform<abi_t>::template test<E>(
                        1 + small, 1 - small, -1, fmop, -small * small);
                    // overflow cancels
                    dpl::test::ternary_transform<abi_t>::template test<E>(
                        dpp::max_value_v<E>, 2, -dpp::max_value_v<E>, fmop,
                        dpp::max_value_v<E>);
                } else if constexpr (fmop == dpp::fmsub) {
                    dpl::test::ternary_transform<abi_t>::template test<E>(
                        1 + small, 1 - small, 1, fmop, -small * small);
                    // overflow cancels
                    dpl::test::ternary_transform<abi_t>::template test<E>(
                        dpp::max_value_v<E>, 2, dpp::max_value_v<E>, fmop,
                        dpp::max_value_v<E>);
                } else if constexpr (fmop == dpp::fnmadd) {
                    dpl::test::ternary_transform<abi_t>::template test<E>(
                        1 + small, 1 - small, 1, fmop, small * small);
                    // overflow cancels
                    dpl::test::ternary_transform<abi_t>::template test<E>(
                        dpp::max_value_v<E>, 2, dpp::max_value_v<E>, fmop,
                        -dpp::max_value_v<E>);
                } else if constexpr (fmop == dpp::fnmsub) {
                    dpl::test::ternary_transform<abi_t>::template test<E>(
                        1 + small, 1 - small, -1, fmop, small * small);
                    // overflow cancels
                    dpl::test::ternary_transform<abi_t>::template test<E>(
                        dpp::max_value_v<E>, 2, -dpp::max_value_v<E>, fmop,
                        -dpp::max_value_v<E>);
                }
            }

        return true;
    }
};
} // namespace dpl::test
