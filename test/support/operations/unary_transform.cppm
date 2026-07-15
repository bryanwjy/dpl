// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
#include <cstdlib>

export module dpl.test:unary_transform;
export import :utils.span;
export import :utils.comparison;

import dpl;

namespace dpl::test {

namespace dpp = dpl::datapar;

// Generic test infrastructure for binary maskable_transform SIMD operations
export template <dpp::simd_abi A>
class unary_transform {

    template <dpp::simd_element_for<A> E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <dpp::simd_element_for<A> E>
    using span_t = conditional_t<dpp::fixed_width_abi<A>,
        span<E const, abi_traits<E>::size>, span<E const>>;

public:
    template <dpp::simd_element_for<A> E, dpp::simd_element_for<A> S>
    constexpr void test(span_t<E> data, auto op, span_t<S> expected_vals,
        auto cmp = dpp::cmpeq) const noexcept {
        if constexpr (dpp::scalable_abi<A>) {
            assert(expected_vals.size() == abi_traits<E>::size());
            assert(data.size() == abi_traits<E>::size());
        }
        auto const arg = dpp::load<E, A>(data.data());
        auto const expected = dpp::load<E, A>(expected_vals.data());
        assert(dpp::all_of(cmp(op(arg), expected)));
    }

    template <dpp::simd_element_for<A> E, dpp::simd_element_for<A> S>
    constexpr void test_masked(span_t<E> data, auto op, S src_val,
        auto cmp = test::bitcmp) const noexcept {
        if constexpr (dpp::scalable_abi<A>) {
            assert(data.size() == abi_traits<E>::size());
        }

        auto const arg = dpp::load<E, A>(data.data());
        auto const src = dpp::broadcast<A>(src_val);
        auto const all_true = dpp::broadcast<E, A>(true);
        auto const all_false = dpp::broadcast<E, A>(false);
        auto const alt_mask = (dpp::lane_index<A, E>() & 1) != 0;
        auto const op_val = op(arg);
        auto const vzero = dpp::broadcast<E, A>(dpp::zero);

        // merge-masked: active → op_val, inactive → src
        {
            assert(dpp::all_of(cmp(op(src, all_true, arg), op_val)));
            assert(dpp::none_of(cmp(op(src, all_false, arg), op_val)));
            assert(dpp::all_of(cmp(op(src, all_false, arg), src)));

            auto const actual = op(src, alt_mask, arg);
            assert(dpp::all_of(cmp(actual, op_val) == alt_mask));
            assert(dpp::all_of(cmp(actual == src) == !alt_mask));
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
            using alt_cmask_t = dpp::const_mask<lanes,
                dpl::to_underlying(bitset_t(0x5555555555555555llu))>;
            using all_cmask_t =
                dpp::const_mask<lanes, dpl::to_underlying(~bitset_t())>;
            using none_cmask_t = dpp::const_mask<lanes, 0>;
            constexpr alt_cmask_t alt_cmask;
            constexpr all_cmask_t all_cmask;
            constexpr none_cmask_t none_cmask;
            // merge-masked: active → op_val, inactive → src
            {
                assert(dpp::all_of(cmp(op(src, all_cmask, arg), op_val)));
                assert(dpp::none_of(cmp(op(src, none_cmask, arg), op_val)));
                assert(dpp::all_of(cmp(op(src, none_cmask, arg), src)));

                auto const actual = op(src, alt_cmask, arg);
                assert(dpp::all_of(cmp(actual, op_val) == alt_mask));
                assert(dpp::all_of(cmp(actual, src) == !alt_mask));
            }

            // zero-masked: op(vzero, mask, arg) == op(dpp::zero, mask,
            // arg)
            {
                assert(dpp::all_of(cmp(
                    op(vzero, all_cmask, arg), op(dpp::zero, all_cmask, arg))));
                assert(dpp::all_of(cmp(op(vzero, none_cmask, arg),
                    op(dpp::zero, none_cmask, arg))));
                assert(dpp::all_of(cmp(
                    op(vzero, alt_cmask, arg), op(dpp::zero, alt_cmask, arg))));
            }

            // zero-masked alias: op(mask, arg) == op(dpp::zero, mask, lhs,
            // rhs)
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
};

} // namespace dpl::test
