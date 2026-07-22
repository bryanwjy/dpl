// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
#include <cstdlib>

export module dpl.test:support.binary_transform;
export import :support.span;
export import :support.comparison;
export import :support.bitset_helpers;

import dpl;

namespace dpl::test {

namespace dpp = dpl::datapar;

// Generic test infrastructure for binary maskable_transform SIMD operations
export template <dpp::simd_abi A>
class binary_transform {

    template <dpp::simd_element_for<A> E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename Op, typename E>
    using op_result = typename invoke_result_t<Op, dpp::basic_vector<E, A>,
        dpp::basic_vector<E, A>>::value_type;
    template <dpp::simd_element_for<A> E>
    using span_t = conditional_t<dpp::fixed_width_abi<A>,
        span<E const, abi_traits<E>::size>, span<E const>>;
    template <dpp::simd_element_for<A> E>
    using vec_t = dpp::basic_vector<E, A>;

    template <typename E, typename Op, typename Cmp>
    static constexpr void test_masked(vec_t<E> const lhs, vec_t<E> const rhs,
        Op const op, vec_t<op_result<Op, E>> const src, Cmp cmp) noexcept {
        auto const all_true = dpp::broadcast<E, A>(true);
        auto const all_false = dpp::broadcast<E, A>(false);
        auto const alt_mask = (dpp::lane_index<A, E>() & 1) == 0;
        auto const vop = op(lhs, rhs);
        auto const vzero = dpp::broadcast<E, A>(dpp::zero);

        // merge-masked: active → vop, inactive → src
        {
            assert(dpp::all_of(cmp(op(src, all_true, lhs, rhs), vop)));
            assert(dpp::none_of(cmp(op(src, all_false, lhs, rhs), vop)));
            assert(dpp::all_of(cmp(op(src, all_false, lhs, rhs), src)));

            auto const actual = op(src, alt_mask, lhs, rhs);
            assert(dpp::all_of(cmp(actual, vop) == alt_mask));
            assert(dpp::all_of(cmp(actual, src) == !alt_mask));
        }

        // zero-masked: op(vzero, mask, lhs, rhs) == op(dpp::zero, mask,
        // lhs, rhs)
        {
            assert(dpp::all_of(cmp(op(vzero, all_true, lhs, rhs),
                op(dpp::zero, all_true, lhs, rhs))));
            assert(dpp::all_of(cmp(op(vzero, all_false, lhs, rhs),
                op(dpp::zero, all_false, lhs, rhs))));
            assert(dpp::all_of(cmp(op(vzero, alt_mask, lhs, rhs),
                op(dpp::zero, alt_mask, lhs, rhs))));
        }

        // zero-masked alias: op(mask, lhs, rhs) == op(dpp::zero, mask, lhs,
        // rhs)
        {
            assert(dpp::all_of(cmp(
                op(all_true, lhs, rhs), op(dpp::zero, all_true, lhs, rhs))));
            assert(dpp::all_of(cmp(
                op(all_false, lhs, rhs), op(dpp::zero, all_false, lhs, rhs))));
            assert(dpp::all_of(cmp(
                op(alt_mask, lhs, rhs), op(dpp::zero, alt_mask, lhs, rhs))));
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto lanes = abi_traits<E>::size();
            using bitset_t = dpl::bitset<lanes>;
            using alt_cmask_t =
                dpp::const_mask<lanes, test::repeat_byte<lanes>(0x55u)>;
            using all_cmask_t =
                dpp::const_mask<lanes, dpl::to_underlying(~bitset_t())>;
            using none_cmask_t = dpp::const_mask<lanes, 0>;
            constexpr alt_cmask_t alt_cmask;
            constexpr all_cmask_t all_cmask;
            constexpr none_cmask_t none_cmask;
            {
                assert(dpp::all_of(cmp(op(src, all_cmask, lhs, rhs), vop)));
                assert(dpp::none_of(cmp(op(src, none_cmask, lhs, rhs), vop)));
                assert(dpp::all_of(cmp(op(src, none_cmask, lhs, rhs), src)));

                auto const actual = op(src, alt_cmask, lhs, rhs);
                assert(dpp::all_of(cmp(actual, vop) == alt_mask));
                assert(dpp::all_of(cmp(actual, src) == !alt_mask));
            }

            {
                assert(dpp::all_of(cmp(op(vzero, all_cmask, lhs, rhs),
                    op(dpp::zero, all_cmask, lhs, rhs))));
                assert(dpp::all_of(cmp(op(vzero, none_cmask, lhs, rhs),
                    op(dpp::zero, none_cmask, lhs, rhs))));
                assert(dpp::all_of(cmp(op(vzero, alt_cmask, lhs, rhs),
                    op(dpp::zero, alt_cmask, lhs, rhs))));
            }

            {
                assert(dpp::all_of(cmp(op(all_cmask, lhs, rhs),
                    op(dpp::zero, all_cmask, lhs, rhs))));
                assert(dpp::all_of(cmp(op(none_cmask, lhs, rhs),
                    op(dpp::zero, none_cmask, lhs, rhs))));
                assert(dpp::all_of(cmp(op(alt_cmask, lhs, rhs),
                    op(dpp::zero, alt_cmask, lhs, rhs))));
            }
        }
    }

public:
    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(dpp::cmpeq)>
    static constexpr bool test(span_t<E> lhs, span_t<E> rhs, Op op,
        span_t<op_result<Op, E>> expected, Cmp cmp = dpp::cmpeq) noexcept {
        if constexpr (dpp::scalable_abi<A>) {
            assert(expected.size() == abi_traits<E>::size());
            assert(lhs.size() == abi_traits<E>::size());
            assert(rhs.size() == abi_traits<E>::size());
        }

        auto const vlhs = dpp::load<E, A>(lhs.data());
        auto const vrhs = dpp::load<E, A>(rhs.data());
        auto const vexpected = dpp::load<E, A>(expected.data());
        auto const vactual = op(vlhs, vrhs);
        assert(dpp::all_of(cmp(vactual, vexpected)));
        return true;
    }

    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(dpp::cmpeq)>
    static constexpr bool test(E lhs, E rhs, Op op, op_result<Op, E> expected,
        Cmp cmp = dpp::cmpeq) noexcept {
        auto const vlhs = dpp::broadcast<E, A>(lhs);
        auto const vrhs = dpp::broadcast<E, A>(rhs);
        auto const vexpected = dpp::broadcast<E, A>(expected);
        auto const vactual = op(vlhs, vrhs);
        assert(dpp::all_of(cmp(vactual, vexpected)));
        return true;
    }

    // Use bitcmp for masked tests
    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(test::bitcmp)>
    static constexpr bool test_masked(span_t<E> lhs, span_t<E> rhs, Op op,
        op_result<Op, E> src, Cmp cmp = test::bitcmp) noexcept {
        if constexpr (dpp::scalable_abi<A>) {
            assert(lhs.size() == abi_traits<E>::size());
            assert(rhs.size() == abi_traits<E>::size());
        }
        test_masked(dpp::load<E, A>(lhs.data()), dpp::load<E, A>(rhs.data()),
            op, dpp::broadcast<A>(src), cmp);
        return true;
    }

    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(test::bitcmp)>
    static constexpr bool test_masked(E lhs, E rhs, Op op, op_result<Op, E> src,
        Cmp cmp = test::bitcmp) noexcept {
        test_masked(dpp::broadcast<E, A>(lhs), dpp::broadcast<E, A>(rhs), op,
            dpp::broadcast<A>(src), cmp);
        return true;
    }
};

} // namespace dpl::test
