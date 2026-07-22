// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
#include <cstdlib>

export module dpl.test:support.unary_transform;
export import :support.span;
export import :support.comparison;
export import :support.bitset_helpers;

import dpl;

namespace dpl::test {

namespace dpp = dpl::datapar;

// Generic test infrastructure for unary maskable_transform SIMD operations
export template <dpp::simd_abi A>
class unary_transform {

    template <dpp::simd_element_for<A> E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename Op, typename E>
    using op_result =
        typename invoke_result_t<Op, dpp::basic_vector<E, A>>::value_type;
    template <dpp::simd_element_for<A> E>
    using span_t = conditional_t<dpp::fixed_width_abi<A>,
        span<E const, abi_traits<E>::size>, span<E const>>;
    template <dpp::simd_element_for<A> E>
    using vec_t = dpp::basic_vector<E, A>;

    template <typename E, typename Op, typename Cmp>
    static constexpr void test_masked(vec_t<E> const arg, Op const op,
        vec_t<op_result<Op, E>> const src, Cmp cmp) noexcept {
        auto const all_true = dpp::broadcast<E, A>(true);
        auto const all_false = dpp::broadcast<E, A>(false);
        auto const alt_mask = (dpp::lane_index<A, E>() & 1) == 0;
        auto const vop = op(arg);
        auto const vzero = dpp::broadcast<E, A>(dpp::zero);

        // merge-masked: active → vop, inactive → src
        {
            assert(dpp::all_of(cmp(op(src, all_true, arg), vop)));
            assert(dpp::none_of(cmp(op(src, all_false, arg), vop)));
            assert(dpp::all_of(cmp(op(src, all_false, arg), src)));

            auto const actual = op(src, alt_mask, arg);
            assert(dpp::all_of(cmp(actual, vop) == alt_mask));
            assert(dpp::all_of(cmp(actual, src) == !alt_mask));
        }

        // zero-masked: op(vzero, mask, arg) == op(dpp::zero, mask,
        // arg)
        {
            assert(dpp::all_of(
                cmp(op(vzero, all_true, arg), op(dpp::zero, all_true, arg))));
            assert(dpp::all_of(
                cmp(op(vzero, all_false, arg), op(dpp::zero, all_false, arg))));
            assert(dpp::all_of(
                cmp(op(vzero, alt_mask, arg), op(dpp::zero, alt_mask, arg))));
        }

        // zero-masked alias: op(mask, arg) == op(dpp::zero, mask, lhs,
        // rhs)
        {
            assert(dpp::all_of(
                cmp(op(all_true, arg), op(dpp::zero, all_true, arg))));
            assert(dpp::all_of(
                cmp(op(all_false, arg), op(dpp::zero, all_false, arg))));
            assert(dpp::all_of(
                cmp(op(alt_mask, arg), op(dpp::zero, alt_mask, arg))));
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto lanes = abi_traits<E>::size();
            using bitset_t = dpl::bitset<lanes>;
            static_assert(lanes <= sizeof(0u) * dpl::char_bit_v);
            using alt_cmask_t =
                dpp::const_mask<lanes, test::repeat_byte<lanes>(0x55u)>;
            using all_cmask_t =
                dpp::const_mask<lanes, dpl::to_underlying(~bitset_t())>;
            using none_cmask_t = dpp::const_mask<lanes, 0>;
            constexpr alt_cmask_t alt_cmask;
            constexpr all_cmask_t all_cmask;
            constexpr none_cmask_t none_cmask;
            {
                assert(dpp::all_of(cmp(op(src, all_cmask, arg), vop)));
                assert(dpp::none_of(cmp(op(src, none_cmask, arg), vop)));
                assert(dpp::all_of(cmp(op(src, none_cmask, arg), src)));

                auto const actual = op(src, alt_cmask, arg);
                assert(dpp::all_of(cmp(actual, vop) == alt_mask));
                assert(dpp::all_of(cmp(actual, src) == !alt_mask));
            }

            {
                assert(dpp::all_of(cmp(
                    op(vzero, all_cmask, arg), op(dpp::zero, all_cmask, arg))));
                assert(dpp::all_of(cmp(op(vzero, none_cmask, arg),
                    op(dpp::zero, none_cmask, arg))));
                assert(dpp::all_of(cmp(
                    op(vzero, alt_cmask, arg), op(dpp::zero, alt_cmask, arg))));
            }

            {
                assert(dpp::all_of(
                    cmp(op(all_cmask, arg), op(dpp::zero, all_cmask, arg))));
                assert(dpp::all_of(
                    cmp(op(none_cmask, arg), op(dpp::zero, none_cmask, arg))));
                assert(dpp::all_of(
                    cmp(op(alt_cmask, arg), op(dpp::zero, alt_cmask, arg))));
            }
        }
    }

public:
    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(dpp::cmpeq)>
    static constexpr bool test(span_t<E> args, Op op,
        span_t<op_result<Op, E>> expected, Cmp cmp = dpp::cmpeq) noexcept {
        if constexpr (dpp::scalable_abi<A>) {
            assert(expected.size() == abi_traits<E>::size());
            assert(args.size() == abi_traits<E>::size());
        }

        auto const vargs = dpp::load<E, A>(args.data());
        auto const vexpected = dpp::load<A>(expected.data());
        return dpp::all_of(cmp(op(vargs), vexpected));
    }

    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(dpp::cmpeq)>
    static constexpr bool test(E arg, Op op, op_result<Op, E> expected,
        Cmp cmp = dpp::cmpeq) noexcept {
        auto const varg = dpp::broadcast<E, A>(arg);
        auto const vexpected = dpp::broadcast<A>(expected);
        return dpp::all_of(cmp(op(varg), vexpected));
    }

    // Use bitcmp for masked tests
    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(test::bitcmp)>
    static constexpr bool test_masked(span_t<E> args, Op op,
        op_result<Op, E> src, Cmp cmp = test::bitcmp) noexcept {
        if constexpr (dpp::scalable_abi<A>) {
            assert(args.size() == abi_traits<E>::size());
        }

        test_masked(
            dpp::load<E, A>(args.data()), op, dpp::broadcast<A>(src), cmp);
        return true;
    }

    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(test::bitcmp)>
    static constexpr bool test_masked(
        E arg, Op op, op_result<Op, E> src, Cmp cmp = test::bitcmp) noexcept {
        test_masked(dpp::broadcast<E, A>(arg), op, dpp::broadcast<A>(src), cmp);
        return true;
    }
};

} // namespace dpl::test
