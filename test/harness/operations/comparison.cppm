// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.comparison;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_primitive_operation auto cmpop, dpp::simd_abi A>
class comparison {
    template <typename E>
    static constexpr bool expected_op(E lhs, E rhs) noexcept {
        if constexpr (cmpop == dpp::cmplt) {
            return lhs < rhs;
        } else if constexpr (cmpop == dpp::cmpgt) {
            return lhs > rhs;
        } else if constexpr (cmpop == dpp::cmple) {
            return lhs <= rhs;
        } else if constexpr (cmpop == dpp::cmpge) {
            return lhs >= rhs;
        } else if constexpr (cmpop == dpp::cmpeq) {
            return lhs == rhs;
        } else {
            static_assert(cmpop == dpp::cmpneq);
            return lhs != rhs;
        }
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return comparison::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        using bitset_t = dpl::bitset<dpp::simd_abi_traits<A, E>::size>;
        test::array_generator<A, E> const data_generator;
        test::bit_generator<dpp::simd_abi_traits<A, E>::size> const
            bit_generator;
        for (auto i = 0zu; i < 5; ++i) {
            auto const lhs = data_generator(engine);
            auto const rhs = [&]() {
                if constexpr (cmpop == dpp::cmpeq || cmpop == dpp::cmpneq) {
                    test::scalar_generator<int> const binary_generator(0, 2);
                    auto result = lhs;
                    for (auto& val : result) {
                        if (binary_generator(engine) == 0) {
                            val = data_generator.scalar(engine);
                        }
                    }

                    return result;
                } else {
                    return data_generator(engine);
                }
            }();

            bitset_t expected;
            for (auto i = 0zu; i < expected.size(); ++i) {
                expected.set(i, expected_op(lhs[i], rhs[i]));
            }

            test::binary_predicate<A>::template test<E>(
                lhs, rhs, cmpop, expected);
            test::binary_predicate<A>::template test_masked<E>(lhs, rhs, cmpop);
        }

        if constexpr (dpl::unsigned_integral<E> &&
            (cmpop == dpp::cmpeq || cmpop == dpp::cmpneq)) {
            // Mask comparisons
            for (auto i = 0zu; i < 5; ++i) {
                auto const lhs = bit_generator(engine);
                auto const rhs = bit_generator(engine);
                auto const vlhs = dpp::from_bitset<A, E>(lhs);
                auto const vrhs = dpp::from_bitset<A, E>(rhs);

                if constexpr (cmpop == dpp::cmpeq) {
                    assert(dpp::to_bitset(cmpop(vlhs, vlhs)) == ~bitset_t());
                    assert(dpp::to_bitset(cmpop(vlhs, dpp::bwnot(vlhs))) ==
                        bitset_t());
                    assert(dpp::to_bitset(cmpop(vlhs, vrhs)) == ~(lhs ^ rhs));
                } else {
                    assert(dpp::to_bitset(cmpop(vlhs, vlhs)) == bitset_t());
                    assert(dpp::to_bitset(cmpop(vlhs, dpp::bwnot(vlhs))) ==
                        ~bitset_t());
                    assert(dpp::to_bitset(cmpop(vlhs, vrhs)) == (lhs ^ rhs));
                }
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            if constexpr (cmpop == dpp::cmpeq || cmpop == dpp::cmpge ||
                cmpop == dpp::cmple) {
                test::binary_predicate<A>::template test<E>(
                    -0.0, 0.0, cmpop, ~bitset_t());
                test::binary_predicate<A>::template test<E>(
                    0.0, -0.0, cmpop, ~bitset_t());
                test::binary_predicate<A>::template test<E>(
                    dpp::infinity, dpp::infinity, cmpop, ~bitset_t());
                test::binary_predicate<A>::template test<E>(
                    -dpp::infinity, -dpp::infinity, cmpop, ~bitset_t());
            } else {
                test::binary_predicate<A>::template test<E>(
                    -0.0, 0.0, cmpop, bitset_t());
                test::binary_predicate<A>::template test<E>(
                    0.0, -0.0, cmpop, bitset_t());
            }

            if (!test::finite_math_only()) {
                auto const finite = data_generator.scalar(engine);

                if constexpr (cmpop == dpp::cmpneq) {
                    test::binary_predicate<A>::template test<E>(
                        finite, dpp::nan_v<E>, cmpop, ~bitset_t());
                    test::binary_predicate<A>::template test<E>(
                        dpp::nan_v<E>, finite, cmpop, ~bitset_t());
                    test::binary_predicate<A>::template test<E>(
                        dpp::nan_v<E>, dpp::nan_v<E>, cmpop, ~bitset_t());
                } else {
                    test::binary_predicate<A>::template test<E>(
                        finite, dpp::nan_v<E>, cmpop, bitset_t());
                    test::binary_predicate<A>::template test<E>(
                        dpp::nan_v<E>, finite, cmpop, bitset_t());
                }

                if constexpr (cmpop == dpp::cmplt || cmpop == dpp::cmple) {

                    test::binary_predicate<A>::template test<E>(
                        dpp::infinity, -dpp::infinity, cmpop, bitset_t());

                    test::binary_predicate<A>::template test<E>(
                        finite, dpp::infinity, cmpop, ~bitset_t());
                    test::binary_predicate<A>::template test<E>(
                        dpp::infinity, finite, cmpop, bitset_t());

                    test::binary_predicate<A>::template test<E>(
                        finite, -dpp::infinity, cmpop, bitset_t());
                    test::binary_predicate<A>::template test<E>(
                        -dpp::infinity, finite, cmpop, ~bitset_t());
                } else if constexpr (cmpop == dpp::cmpgt ||
                    cmpop == dpp::cmpge) {
                    test::binary_predicate<A>::template test<E>(
                        dpp::infinity, -dpp::infinity, cmpop, ~bitset_t());
                    test::binary_predicate<A>::template test<E>(
                        -dpp::infinity, dpp::infinity, cmpop, bitset_t());

                    test::binary_predicate<A>::template test<E>(
                        finite, dpp::infinity, cmpop, bitset_t());
                    test::binary_predicate<A>::template test<E>(
                        dpp::infinity, finite, cmpop, ~bitset_t());

                    test::binary_predicate<A>::template test<E>(
                        finite, -dpp::infinity, cmpop, ~bitset_t());
                    test::binary_predicate<A>::template test<E>(
                        -dpp::infinity, finite, cmpop, bitset_t());
                } else if constexpr (cmpop == dpp::cmpneq) {
                    test::binary_predicate<A>::template test<E>(
                        dpp::infinity, -dpp::infinity, cmpop, ~bitset_t());
                    test::binary_predicate<A>::template test<E>(
                        -dpp::infinity, dpp::infinity, cmpop, ~bitset_t());
                }
            }
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using eq_comparison = comparison<dpp::cmpeq, A>;
export template <dpp::simd_abi A>
using neq_comparison = comparison<dpp::cmpneq, A>;
export template <dpp::simd_abi A>
using lt_comparison = comparison<dpp::cmplt, A>;
export template <dpp::simd_abi A>
using le_comparison = comparison<dpp::cmple, A>;
export template <dpp::simd_abi A>
using gt_comparison = comparison<dpp::cmpgt, A>;
export template <dpp::simd_abi A>
using ge_comparison = comparison<dpp::cmpge, A>;
} // namespace dpl::test
