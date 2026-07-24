// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.division;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class division {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept {
        return lhs / rhs;
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return division::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    requires dpl::floating_point_like<E>
    static constexpr bool run(Rng& engine) {
        constexpr auto max = []() {
            constexpr auto bias = dpl::floating_point_traits<E>::exponent_bias;
            constexpr auto exp = dpl::floating_point_traits<E>::exponent_mask;
            constexpr auto shift = dpl::countr_zero(exp);
            // minus 1 for reduced range, so that sqrt * sqrt < max_value
            constexpr auto sqrt =
                dpl::bitset<dpl::type_bit_v<E>>(
                    ((dpl::to_underlying(exp >> shift) - bias) >> 1) + bias - 1)
                << shift;
            return dpl::bit_cast<E>(sqrt);
        }();

        constexpr auto min = -max;

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
            lhs, rhs, dpp::divide, expected, test::bitcmp);
        test::binary_transform<abi_t>::template test_masked<E>(
            lhs, rhs, dpp::divide, src);

        auto const rnd = src_generator(engine);
        test::binary_transform<abi_t>::template test<E>(0, rnd, dpp::divide, 0);
        test::binary_transform<abi_t>::template test<E>(
            rnd, 1, dpp::divide, rnd);
        test::binary_transform<abi_t>::template test<E>(
            rnd, -1, dpp::divide, -rnd);

        if (test::ieee_denormal()) {
            using ubit_t = dpp::unsigned_representation_t<E>;
            auto const denorm = dpl::bit_cast<E>(ubit_t(1));
            test::binary_transform<abi_t>::template test<E>(
                1, denorm, dpp::divide, dpp::infinity_v<E>);
            test::binary_transform<abi_t>::template test<E>(
                dpl::bit_cast<E>(ubit_t(2)), 2.0, dpp::divide, denorm);
        }

        if (!test::finite_math_only()) {
            constexpr auto inf = dpp::infinity_v<E>;

            test::binary_transform<abi_t>::template test<E>(
                inf, 2.0, dpp::divide, inf);
            test::binary_transform<abi_t>::template test<E>(
                inf, -2.0, dpp::divide, -inf);
            test::binary_transform<abi_t>::template test<E>(
                -inf, 2.0, dpp::divide, -inf);
            test::binary_transform<abi_t>::template test<E>(
                -inf, -2.0, dpp::divide, inf);

            test::binary_transform<abi_t>::template test<E>(
                2.0, inf, dpp::divide, 0.0, test::bitcmp);
            test::binary_transform<abi_t>::template test<E>(
                2.0, -inf, dpp::divide, -0.0, test::bitcmp);
            test::binary_transform<abi_t>::template test<E>(
                -2.0, inf, dpp::divide, -0.0, test::bitcmp);
            test::binary_transform<abi_t>::template test<E>(
                -2.0, -inf, dpp::divide, 0.0, test::bitcmp);

            if not consteval {
                test::binary_transform<abi_t>::template test<E>(
                    1, 0, dpp::divide, inf);
                test::binary_transform<abi_t>::template test<E>(
                    -1, 0, dpp::divide, -inf);

                auto const cmpunord = [](auto lhs, auto rhs) {
                    return dpp::isnan(lhs) && dpp::isnan(rhs);
                };
                test::binary_transform<abi_t>::template test<E>(
                    inf, inf, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<abi_t>::template test<E>(
                    inf, -inf, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<abi_t>::template test<E>(
                    -inf, inf, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<abi_t>::template test<E>(
                    -inf, -inf, dpp::divide, dpp::nan, cmpunord);

                test::binary_transform<abi_t>::template test<E>(
                    0.0, 0.0, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<abi_t>::template test<E>(
                    -0.0, 0.0, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<abi_t>::template test<E>(
                    0.0, -0.0, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<abi_t>::template test<E>(
                    -0.0, -0.0, dpp::divide, dpp::nan, cmpunord);

                test::binary_transform<abi_t>::template test<E>(
                    dpp::nan, 1, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<abi_t>::template test<E>(
                    1, dpp::nan, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<abi_t>::template test<E>(
                    inf, dpp::nan, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<abi_t>::template test<E>(
                    dpp::nan, inf, dpp::divide, dpp::nan, cmpunord);
            }
        }

        return true;
    }
};
} // namespace dpl::test
