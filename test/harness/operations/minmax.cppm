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
        test::array_generator<A, E> const data_generator(test::half_range);
        test::scalar_generator<E> const src_generator(
            dpp::max_value_v<E> / 4 * 3, dpp::max_value_v<E>);
        auto const lhs = data_generator(engine);
        auto const rhs = data_generator(engine);
        auto const src = src_generator(engine);

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op(lhs[i], rhs[i]);
        }

        test::binary_transform<A>::template test<E>(lhs, rhs, op, expected);
        test::binary_transform<A>::template test_masked<E>(lhs, rhs, op, src);

        // May not guarantee IEEE behaviour if nan or -0.0 on some
        // architectures
        auto const strict_tests = [&] {
            if constexpr (dpl::floating_point_like<E>) {
                if constexpr (op == dpp::min) {
                    test::binary_transform<A>::template test<E>(
                        -0.0, 0.0, op, -0.0, test::bitcmp);
                    test::binary_transform<A>::template test<E>(
                        0.0, -0.0, op, -0.0, test::bitcmp);
                } else {
                    test::binary_transform<A>::template test<E>(
                        -0.0, 0.0, op, 0.0, test::bitcmp);
                    test::binary_transform<A>::template test<E>(
                        0.0, -0.0, op, 0.0, test::bitcmp);
                }

                if (!test::finite_math_only()) {
                    auto const finite = src_generator(engine);
                    test::binary_transform<A>::template test<E>(
                        finite, dpp::nan_v<E>, op, finite);
                    test::binary_transform<A>::template test<E>(
                        dpp::nan_v<E>, finite, op, finite);
                }
            }
        };

        if not consteval {
            if constexpr (!is_strict_minmax<A, E>) {
                return true;
            }
        }

        strict_tests();
        return true;
    }
};

export template <dpp::simd_abi A>
using minimization = minmax<dpp::min, A>;
export template <dpp::simd_abi A>
using maximization = minmax<dpp::max, A>;

} // namespace dpl::test
