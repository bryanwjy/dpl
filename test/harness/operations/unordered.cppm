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
    static constexpr bool run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        static_assert(dpl::floating_point_like<E>);

        using bitset_t = dpl::bitset<dpp::simd_abi_traits<A, E>::size>;
        test::array_generator<A, E> const data_generator;

        for (auto i = 0zu; i < 5; ++i) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            constexpr bitset_t expected;
            test::binary_predicate<A>::template test<E>(
                lhs, rhs, dpp::cmpunord, expected);
        }

        if (test::finite_math_only()) {
            return true;
        }

        for (auto i = 0zu; i < 2; ++i) {
            auto const lhs = data_generator(engine);
            auto const rhs = [&]() {
                test::scalar_generator<int> const binary_generator(0, 2);
                auto result = lhs;
                for (auto& val : result) {
                    if (binary_generator(engine) == 0) {
                        val = dpp::nan;
                    }
                }

                return result;
            }();

            bitset_t expected;
            for (auto i = 0zu; i < expected.size(); ++i) {
                expected.set(i, expected_op(lhs[i], rhs[i]));
            }

            test::binary_predicate<A>::template test<E>(
                lhs, rhs, dpp::cmpunord, expected);
            test::binary_predicate<A>::template test<E>(
                rhs, lhs, dpp::cmpunord, expected);
            test::binary_predicate<A>::template test_masked<E>(
                lhs, rhs, dpp::cmpunord);
            test::binary_predicate<A>::template test_masked<E>(
                rhs, lhs, dpp::cmpunord);
        }

        constexpr bitset_t ordered;
        constexpr bitset_t unordered = ~ordered;

        test::binary_predicate<A>::template test<E>(
            -0.0, 0.0, dpp::cmpunord, ordered);
        test::binary_predicate<A>::template test<E>(
            0.0, -0.0, dpp::cmpunord, ordered);
        test::binary_predicate<A>::template test<E>(
            dpp::infinity, dpp::infinity, dpp::cmpunord, ordered);
        test::binary_predicate<A>::template test<E>(
            -dpp::infinity, -dpp::infinity, dpp::cmpunord, ordered);

        {
            auto const finite = data_generator.scalar(engine);
            test::binary_predicate<A>::template test<E>(
                dpp::infinity, finite, dpp::cmpunord, ordered);
            test::binary_predicate<A>::template test<E>(
                -dpp::infinity, finite, dpp::cmpunord, ordered);
            test::binary_predicate<A>::template test<E>(
                finite, dpp::infinity, dpp::cmpunord, ordered);
            test::binary_predicate<A>::template test<E>(
                finite, -dpp::infinity, dpp::cmpunord, ordered);

            test::binary_predicate<A>::template test<E>(
                finite, dpp::nan_v<E>, dpp::cmpunord, unordered);
            test::binary_predicate<A>::template test<E>(
                dpp::nan_v<E>, finite, dpp::cmpunord, unordered);
            test::binary_predicate<A>::template test<E>(
                dpp::nan_v<E>, dpp::nan_v<E>, dpp::cmpunord, unordered);
        }

        return true;
    }
};

} // namespace dpl::test
