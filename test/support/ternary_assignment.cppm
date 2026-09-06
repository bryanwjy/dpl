// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.support:ternary_assignment;
import :span;
import :comparison;
import :bitset_helpers;
import :ternary_transform;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;
inline namespace support {
export template <dpp::simd_abi A>
class ternary_assignment {

    template <dpp::simd_element_for<A> E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <dpp::simd_element_for<A> E>
    using span_t = conditional_t<dpp::fixed_width_abi<A>,
        span<E const, abi_traits<E>::size>, span<E const>>;
    template <dpp::simd_element_for<A> E>
    using vec_t = dpp::basic_vector<E, A>;

    template <typename E, typename Op, typename Cmp>
    static constexpr void test_masked(vec_t<E> const lhs, vec_t<E> const mid,
        vec_t<E> const rhs, Op const op, vec_t<E> const src, Cmp cmp) noexcept {
        auto const all_true = dpp::broadcast<E, A>(true);
        auto const all_false = dpp::broadcast<E, A>(false);
        auto const alt_mask =
            dpp::cmpeq(dpp::bwand(dpp::lane_index<A, E>(), 1), dpp::zero);
        auto const vop = op(lhs, mid, rhs);
        auto const vzero = dpp::broadcast<E, A>(dpp::zero);
        // merge-masked: active → vop, inactive → src
        {
            assert(dpp::all_of(cmp(op(lhs, all_true, mid, rhs), vop)));
            assert(dpp::all_of(cmp(op(src, all_false, mid, rhs), src)));
            {
                auto const actual = op(src, all_false, mid, rhs);
                assert(
                    dpp::all_of(dpp::cmpeq(cmp(actual, vop), cmp(vop, src))));
            }

            {
                auto const dst = dpp::select(alt_mask, lhs, src);
                auto const actual = op(dst, alt_mask, mid, rhs);
                assert(dpp::all_of(dpp::cmpeq(
                    dpp::bwand(cmp(actual, vop), alt_mask), alt_mask)));
                assert(dpp::all_of(dpp::cmpneq(
                    dpp::bwandnot(cmp(actual, src), alt_mask), alt_mask)));
            }
        }

        // zero-masked: op(vzero, mask, lhs, mid, rhs) == op(dpp::zero, mask,
        // lhs, mid, rhs)
        {
            assert(dpp::all_of(cmp(op(lhs, all_true, mid, rhs),
                op(dpp::zero, all_true, lhs, mid, rhs))));
            assert(dpp::all_of(cmp(op(vzero, all_false, mid, rhs),
                op(dpp::zero, all_false, lhs, mid, rhs))));
            auto const dst = dpp::select(alt_mask, lhs, dpp::zero);
            assert(dpp::all_of(cmp(op(dst, alt_mask, mid, rhs),
                op(dpp::zero, alt_mask, lhs, mid, rhs))));
        }

        // zero-masked alias: op(mask, lhs, mid, rhs) == op(dpp::zero, mask,
        // lhs, rhs)
        {
            assert(dpp::all_of(cmp(op(all_true, lhs, mid, rhs),
                op(dpp::zero, all_true, lhs, mid, rhs))));
            assert(dpp::all_of(cmp(op(all_false, lhs, mid, rhs),
                op(dpp::zero, all_false, lhs, mid, rhs))));
            assert(dpp::all_of(cmp(op(alt_mask, lhs, mid, rhs),
                op(dpp::zero, alt_mask, lhs, mid, rhs))));
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
                assert(dpp::all_of(cmp(op(lhs, all_cmask, mid, rhs), vop)));
                assert(dpp::all_of(cmp(op(src, none_cmask, mid, rhs), src)));
                {
                    auto const actual = op(src, none_cmask, mid, rhs);
                    assert(dpp::all_of(
                        dpp::cmpeq(cmp(actual, vop), cmp(vop, src))));
                }

                {
                    auto const dst = dpp::select(alt_cmask, lhs, src);
                    auto const actual = op(dst, alt_cmask, mid, rhs);
                    assert(dpp::all_of(dpp::cmpeq(
                        dpp::bwand(cmp(actual, vop), alt_mask), alt_mask)));
                    assert(dpp::all_of(dpp::cmpneq(
                        dpp::bwandnot(cmp(actual, src), alt_mask), alt_mask)));
                }
            }

            {
                assert(dpp::all_of(cmp(op(lhs, all_cmask, mid, rhs),
                    op(dpp::zero, all_cmask, lhs, mid, rhs))));
                assert(dpp::all_of(cmp(op(vzero, none_cmask, mid, rhs),
                    op(dpp::zero, none_cmask, lhs, mid, rhs))));
                auto const dst = dpp::select(alt_cmask, lhs, dpp::zero);
                assert(dpp::all_of(cmp(op(dst, alt_cmask, mid, rhs),
                    op(dpp::zero, alt_cmask, lhs, mid, rhs))));
            }

            {
                assert(dpp::all_of(cmp(op(all_cmask, lhs, mid, rhs),
                    op(dpp::zero, all_cmask, lhs, mid, rhs))));
                assert(dpp::all_of(cmp(op(none_cmask, lhs, mid, rhs),
                    op(dpp::zero, none_cmask, lhs, mid, rhs))));
                assert(dpp::all_of(cmp(op(alt_cmask, lhs, mid, rhs),
                    op(dpp::zero, alt_cmask, lhs, mid, rhs))));
            }
        }
    }

public:
    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(dpp::cmpeq)>
    static constexpr bool test(span_t<E> lhs, span_t<E> mid, span_t<E> rhs,
        Op op, span_t<E> expected, Cmp cmp = dpp::cmpeq) noexcept {
        return ternary_transform<A>::template test<E>(
            lhs, mid, rhs, op, expected, cmp);
    }

    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(dpp::cmpeq)>
    static constexpr bool test(
        E lhs, E mid, E rhs, Op op, E expected, Cmp cmp = dpp::cmpeq) noexcept {
        return ternary_transform<A>::template test<E>(
            lhs, mid, rhs, op, expected, cmp);
    }

    // Use bitcmp for masked tests
    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(test::bitcmp)>
    static constexpr bool test_masked(span_t<E> lhs, span_t<E> mid,
        span_t<E> rhs, Op op, E src, Cmp cmp = test::bitcmp) noexcept {
        if constexpr (dpp::scalable_abi<A>) {
            assert(lhs.size() == abi_traits<E>::size());
            assert(mid.size() == abi_traits<E>::size());
            assert(rhs.size() == abi_traits<E>::size());
        }

        test_masked(dpp::load<E, A>(lhs.data()), dpp::load<E, A>(mid.data()),
            dpp::load<E, A>(rhs.data()), op, dpp::broadcast<A>(src), cmp);
        return true;
    }

    template <dpp::simd_element_for<A> E, typename Op,
        typename Cmp = decltype(test::bitcmp)>
    static constexpr bool test_masked(
        E lhs, E mid, E rhs, Op op, E src, Cmp cmp = test::bitcmp) noexcept {
        test_masked(dpp::broadcast<E, A>(lhs), dpp::broadcast<E, A>(mid),
            dpp::broadcast<E, A>(rhs), op, dpp::broadcast<A>(src), cmp);
        return true;
    }
};
} // namespace support
} // namespace dpl::test
