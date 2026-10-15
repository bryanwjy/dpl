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
    using abi_traits = dpp::simd_abi_traits<A, E>;

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
                return comparison::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        constexpr auto lanes = abi_traits<E>::size();
        using bitset_t = dpl::bitset<lanes>;

        test::array_generator<A, E> const data_generator;
        test::scalar_generator<int> const binary_generator(0, 2);
        test::mask_generator<A, E> const mask_generator;
        auto const mtrue = dpp::broadcast<A, E>(true);
        auto const mfalse = dpp::broadcast<A, E>(false);

        for (auto const _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const rhs = [&]() {
                if constexpr (cmpop == dpp::cmpeq || cmpop == dpp::cmpneq) {
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

            auto const expected = [&]() {
                auto aexpected = lhs;
                for (auto const i : linear_counter(aexpected)) {
                    // zero is easier to check for FP
                    aexpected[i] = expected_op(lhs[i], rhs[i])
                        ? dpp::zero_v<E>
                        : dpp::all_bits_v<E>;
                }
                return dpp::cmpeq(dpp::load<A>(aexpected.data()), dpp::zero);
            }();

            test::operation_fixture<A>::test(expected, cmpop, lhs, rhs);
            auto const mask = mask_generator(engine);
            test::operation_fixture<A>::test_masked(cmpop, mask, lhs, rhs);
            test::operation_fixture<A>::test_masked(cmpop, mtrue, lhs, rhs);
            test::operation_fixture<A>::test_masked(cmpop, mfalse, lhs, rhs);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const src = data_generator(engine);

            run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(cmpop, cmask, lhs, rhs);
            });
        }

        if constexpr (dpl::unsigned_integral<E> &&
            (cmpop == dpp::cmpeq || cmpop == dpp::cmpneq)) {
            // Mask comparisons
            auto const vzero = dpp::broadcast<E, A>(dpp::zero);
            for (auto const _ : test_count<E>()) {
                auto const mlhs = mask_generator(engine);
                auto const mrhs = mask_generator(engine);
                auto const vexpected = [&]() {
                    auto alhs = data_generator(engine);
                    auto arhs = alhs;
                    dpp::store(
                        dpp::select(mlhs, dpp::all_bits, vzero), alhs.data());
                    dpp::store(
                        dpp::select(mrhs, dpp::all_bits, vzero), arhs.data());

                    for (auto const i : linear_counter(alhs)) {
                        if constexpr (cmpop == dpp::cmpeq) {
                            alhs[i] = alhs[i] == arhs[i] ? dpp::all_bits_v<E>
                                                         : dpp::zero_v<E>;
                        } else {
                            alhs[i] = alhs[i] != arhs[i] ? dpp::all_bits_v<E>
                                                         : dpp::zero_v<E>;
                        }
                    }

                    return dpp::load<A>(alhs.data());
                }();

                if constexpr (cmpop == dpp::cmpeq) {
                    assert(dpp::all_of(cmpop(mlhs, mlhs)));
                    assert(dpp::none_of(cmpop(mlhs, dpp::bwnot(mlhs))));
                    auto const vactual =
                        dpp::select(cmpop(mlhs, mrhs), dpp::all_bits, vzero);
                    assert(dpp::all_of(dpp::cmpeq(vactual, vexpected)));
                } else {
                    assert(dpp::none_of(cmpop(mlhs, mlhs)));
                    assert(dpp::all_of(cmpop(mlhs, dpp::bwnot(mlhs))));
                    auto const vactual =
                        dpp::select(cmpop(mlhs, mrhs), dpp::all_bits, vzero);
                    assert(dpp::all_of(dpp::cmpeq(vactual, vexpected)));
                }
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            if constexpr (cmpop == dpp::cmpeq || cmpop == dpp::cmpge ||
                cmpop == dpp::cmple) {
                test::operation_fixture<A>::test(
                    mtrue, cmpop, test::splat<E>(-0.0), 0.0);
                test::operation_fixture<A>::test(
                    mtrue, cmpop, 0.0, test::splat<E>(-0.0));
                test::operation_fixture<A>::test(
                    mtrue, cmpop, dpp::infinity, test::splat<E>(dpp::infinity));
                test::operation_fixture<A>::test(mtrue, cmpop, -dpp::infinity,
                    test::splat<E>(-dpp::infinity));
            } else {
                test::operation_fixture<A>::test(
                    mfalse, cmpop, test::splat<E>(-0.0), 0.0);
                test::operation_fixture<A>::test(
                    mfalse, cmpop, 0.0, test::splat<E>(-0.0));
            }

            if (!test::finite_math_only()) {
                auto const finite = data_generator(engine);

                if constexpr (cmpop == dpp::cmpneq) {
                    test::operation_fixture<A>::test(
                        mtrue, cmpop, finite, dpp::nan);
                    test::operation_fixture<A>::test(
                        mtrue, cmpop, dpp::nan, finite);
                    test::operation_fixture<A>::test(
                        mtrue, cmpop, test::splat<E>(dpp::nan), dpp::nan);
                } else {
                    test::operation_fixture<A>::test(
                        mfalse, cmpop, finite, dpp::nan);
                    test::operation_fixture<A>::test(
                        mfalse, cmpop, dpp::nan, finite);
                }

                if constexpr (cmpop == dpp::cmplt || cmpop == dpp::cmple) {
                    test::operation_fixture<A>::test(mfalse, cmpop,
                        dpp::infinity, test::splat<E>(-dpp::infinity));

                    test::operation_fixture<A>::test(
                        mtrue, cmpop, finite, dpp::infinity);
                    test::operation_fixture<A>::test(
                        mfalse, cmpop, dpp::infinity, finite);

                    test::operation_fixture<A>::test(
                        mfalse, cmpop, finite, -dpp::infinity);
                    test::operation_fixture<A>::test(
                        mtrue, cmpop, -dpp::infinity, finite);
                } else if constexpr (cmpop == dpp::cmpgt ||
                    cmpop == dpp::cmpge) {
                    test::operation_fixture<A>::test(mtrue, cmpop,
                        dpp::infinity, test::splat<E>(-dpp::infinity));
                    test::operation_fixture<A>::test(mfalse, cmpop,
                        test::splat<E>(-dpp::infinity), dpp::infinity);

                    test::operation_fixture<A>::test(
                        mfalse, cmpop, finite, dpp::infinity);
                    test::operation_fixture<A>::test(
                        mtrue, cmpop, dpp::infinity, finite);

                    test::operation_fixture<A>::test(
                        mtrue, cmpop, finite, -dpp::infinity);
                    test::operation_fixture<A>::test(
                        mfalse, cmpop, -dpp::infinity, finite);
                } else if constexpr (cmpop == dpp::cmpneq) {
                    test::operation_fixture<A>::test(mtrue, cmpop,
                        dpp::infinity, test::splat<E>(-dpp::infinity));
                    test::operation_fixture<A>::test(mtrue, cmpop,
                        test::splat<E>(-dpp::infinity), dpp::infinity);
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
