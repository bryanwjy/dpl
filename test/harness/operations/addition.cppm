// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.addition;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class addition {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept {
        if constexpr (dpl::signed_integral<E>) {
            return dpl::to_signed(
                dpl::to_unsigned(lhs) + dpl::to_unsigned(rhs));
        } else {
            return lhs + rhs;
        }
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return addition::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<abi_t, E> const data_generator(test::half_range);
        test::scalar_generator<E> const src_generator(
            dpp::max_value_v<E> / 4 * 3, dpp::max_value_v<E>);
        auto const lhs = data_generator(engine);
        auto const rhs = data_generator(engine);
        auto const src = src_generator(engine);

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op(lhs[i], rhs[i]);
        }

        test::binary_transform<abi_t>::template test<E>(
            lhs, rhs, dpp::add, expected, test::bitcmp);
        test::binary_transform<abi_t>::template test_masked<E>(
            lhs, rhs, dpp::add, src);

        if constexpr (dpl::integral<E>) {
            test::binary_transform<abi_t>::template test<E>(0, 0, dpp::add, 0);
            auto const rnd = src_generator(engine);
            test::binary_transform<abi_t>::template test<E>(
                rnd, 0, dpp::add, rnd);
            test::binary_transform<abi_t>::template test<E>(
                0, rnd, dpp::add, rnd);

            constexpr auto max = dpl::integral_traits<E>::max_value;
            constexpr auto min = dpl::integral_traits<E>::min_value;

            auto const wraparound_test = []() {
                if constexpr (dpl::unsigned_integral<E>) {
                    return true;
                } else if not consteval {
                    return true;
                } else {
                    return false;
                }
            }();
            if (wraparound_test) {
                test::binary_transform<abi_t>::template test<E>(
                    max, 1, dpp::add, min);
                test::binary_transform<abi_t>::template test<E>(
                    min, -1, dpp::add, max);
                test::binary_transform<abi_t>::template test<E>(
                    max, max, dpp::add, max - 1);

                test::binary_transform<abi_t>::template test<E>(
                    min, max, dpp::add, static_cast<E>(-1));
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            test::binary_transform<abi_t>::template test<E>(
                -0.0, 0.0, dpp::add, 0.0);
            test::binary_transform<abi_t>::template test<E>(
                0.0, 0.0, dpp::add, 0.0);
            test::binary_transform<abi_t>::template test<E>(
                -0.0, -0.0, dpp::add, 0.0);
            test::binary_transform<abi_t>::template test<E>(
                1.0, -1.0, dpp::add, 0.0);

            if (!test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                test::binary_transform<abi_t>::template test<E>(
                    inf, 1.0, dpp::add, inf);
                test::binary_transform<abi_t>::template test<E>(
                    -inf, -1.0, dpp::add, -inf);
                if not consteval {
                    // nan producing arithmetic is not allowed at constexpr
                    test::binary_transform<abi_t>::template test<E>(inf, -inf,
                        dpp::add, dpp::nan_v<E>, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                    test::binary_transform<abi_t>::template test<E>(dpp::nan,
                        1.0, dpp::add, dpp::nan, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });

                    test::binary_transform<abi_t>::template test<E>(1.0,
                        dpp::nan, dpp::add, dpp::nan, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                }
            }
        }

        return true;
    }
};
} // namespace dpl::test
