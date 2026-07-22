// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test:multiplication;
export import :support.binary_transform;
export import :support.span;
export import :support.comparison;
export import :support.bitset_helpers;
export import :support.data_generator;
export import :support.math_check;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class multiplication {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept {
        return lhs * rhs;
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return multiplication::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        constexpr auto max = []() {
            if constexpr (dpl::integral<E>) {
                constexpr auto shift = dpl::type_bit_v<E> / 2;
                return dpl::integral_traits<E>::max_value >> shift;
            } else {
                constexpr auto bias =
                    dpl::floating_point_traits<E>::exponent_bias;
                constexpr auto exp =
                    dpl::floating_point_traits<E>::exponent_mask;
                constexpr auto shift = dpl::countr_zero(exp);
                // minus 1 for reduced range, so that sqrt * sqrt < max_value
                constexpr auto sqrt =
                    dpl::bitset<dpl::type_bit_v<E>>(
                        ((dpl::to_underlying(exp >> shift) - bias) >> 1) +
                        bias - 1)
                    << shift;
                return dpl::bit_cast<E>(sqrt);
            }
        }();

        constexpr auto min = [max]() {
            if constexpr (dpl::integral<E>) {
                constexpr auto shift = dpl::type_bit_v<E> / 2;
                return dpl::integral_traits<E>::min_value >> shift;
            } else {
                return -max;
            }
        }();

        test::array_generator<abi_t, E> const data_generator(min, max);
        test::scalar_generator<E> const src_generator(
            max * max, dpp::max_value_v<E>);
        auto const lhs = data_generator(engine);
        auto const rhs = data_generator(engine);
        auto const src = src_generator(engine);

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op(lhs[i], rhs[i]);
        }

        test::binary_transform<abi_t>::template test<E>(
            lhs, rhs, dpp::multiply, expected);
        test::binary_transform<abi_t>::template test_masked<E>(
            lhs, rhs, dpp::multiply, src);

        auto const rnd = src_generator(engine);
        test::binary_transform<abi_t>::template test<E>(
            rnd, 1, dpp::multiply, rnd);
        test::binary_transform<abi_t>::template test<E>(
            1, rnd, dpp::multiply, rnd);
        test::binary_transform<abi_t>::template test<E>(
            0, rnd, dpp::multiply, 0);
        test::binary_transform<abi_t>::template test<E>(
            rnd, 0, dpp::multiply, 0);

        if constexpr (dpl::integral<E>) {
            if not consteval {
                constexpr auto int_min = dpl::integral_traits<E>::min_value;
                constexpr auto int_max = dpl::integral_traits<E>::max_value;
                test::binary_transform<abi_t>::template test<E>(
                    int_max, 2, dpp::multiply, static_cast<E>(-2));

                if constexpr (dpl::signed_integral<E>) {
                    test::binary_transform<abi_t>::template test<E>(
                        int_min, 2, dpp::multiply, 0);
                }

                test::binary_transform<abi_t>::template test<E>(
                    int_max, int_max, dpp::multiply, 1);
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            test::binary_transform<abi_t>::template test<E>(
                -0.0, 1.0, dpp::multiply, 0.0);
            test::binary_transform<abi_t>::template test<E>(
                0.0, 1.0, dpp::multiply, 0.0);
            test::binary_transform<abi_t>::template test<E>(
                -0.0, -0.0, dpp::multiply, 0.0);
            test::binary_transform<abi_t>::template test<E>(
                1.0, -1.0, dpp::multiply, -1.0);

            if (test::ieee_denormal()) {
                using ubit_t = dpp::unsigned_representation_t<E>;
                auto const denorm = dpl::bit_cast<E>(ubit_t(1));
                test::binary_transform<abi_t>::template test<E>(
                    denorm, denorm, dpp::multiply, 0.0);
                test::binary_transform<abi_t>::template test<E>(
                    denorm, 2.0, dpp::multiply, dpl::bit_cast<E>(ubit_t(2)));
            }

            if (!test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                test::binary_transform<abi_t>::template test<E>(
                    inf, 2.0, dpp::multiply, inf);
                test::binary_transform<abi_t>::template test<E>(
                    inf, -2.0, dpp::multiply, -inf);
                test::binary_transform<abi_t>::template test<E>(
                    -inf, 2.0, dpp::multiply, -inf);
                test::binary_transform<abi_t>::template test<E>(
                    -inf, -2.0, dpp::multiply, inf);

                test::binary_transform<abi_t>::template test<E>(
                    inf, inf, dpp::multiply, inf);
                test::binary_transform<abi_t>::template test<E>(
                    inf, -inf, dpp::multiply, -inf);
                test::binary_transform<abi_t>::template test<E>(
                    -inf, -inf, dpp::multiply, inf);

                test::binary_transform<abi_t>::template test<E>(
                    dpp::max_value_v<E>, dpp::max_value_v<E>, dpp::multiply,
                    inf);
                test::binary_transform<abi_t>::template test<E>(
                    dpp::max_value_v<E>, -dpp::max_value_v<E>, dpp::multiply,
                    -inf);
            }
        }

        return true;
    }
};
} // namespace dpl::test
