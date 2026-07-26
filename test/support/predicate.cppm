// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.support:predicate;
import :span;
import :comparison;
import :bitset_helpers;

import dpl;

namespace dpl::test {

namespace dpp = dpl::datapar;
inline namespace support {

export template <dpp::simd_abi A>
class unary_predicate {
    template <dpp::simd_element_for<A> E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using mask_t = dpp::basic_mask<E, A>;
    template <dpp::simd_element_for<A> E>
    using span_t = conditional_t<dpp::fixed_width_abi<A>,
        span<E const, abi_traits<E>::size>, span<E const>>;
    template <dpp::simd_element_for<A> E>
    using vec_t = dpp::basic_vector<E, A>;
    template <dpp::simd_element_for<A> E>
    using bitset_t = dpl::bitset<abi_traits<E>::size>;

    template <typename E>
    static constexpr bool all_equal(mask_t<E> lhs, mask_t<E> rhs) noexcept {
        return dpp::to_bitset(lhs) == dpp::to_bitset(rhs);
    }

    template <typename E, typename Op>
    static constexpr void test_masked(
        vec_t<E> const arg, Op const op) noexcept {
        auto const all_true = dpp::broadcast<E, A>(true);
        auto const all_false = dpp::broadcast<E, A>(false);
        auto const alt_mask =
            dpp::cmpeq(dpp::bwand(dpp::lane_index<A, E>(), 1), dpp::zero);
        auto const vop = op(arg);

        // merge-masked: active → vop, inactive → src
        {
            assert(all_equal<E>(op(all_true, arg), all_true));
            assert(all_equal<E>(op(all_false, arg), all_false));
            assert(all_equal<E>(op(alt_mask, arg), dpp::bwand(vop, alt_mask)));
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
                assert(all_equal<E>(op(all_cmask, arg), all_true));
                assert(all_equal<E>(op(none_cmask, arg), all_false));
                assert(all_equal<E>(op(alt_cmask, arg), op(alt_mask, arg)));
            }
        }
    }

public:
    template <dpp::simd_element_for<A> E, typename Op>
    static constexpr bool test(
        span_t<E> args, Op op, bitset_t<E> expected) noexcept
    requires dpp::fixed_width_abi<A>
    {
        auto const vargs = dpp::load<A>(args.data());
        auto const vactual = op(vargs);
        assert(dpp::to_bitset(vactual) == expected);
        return true;
    }

    template <dpp::simd_element_for<A> E, typename Op>
    static constexpr bool test(E arg, Op op, bitset_t<E> expected) noexcept {
        auto const varg = dpp::broadcast<A>(arg);
        auto const vactual = op(varg);
        assert(dpp::to_bitset(vactual) == expected);
        return true;
    }

    // Use bitcmp for masked tests
    template <dpp::simd_element_for<A> E, typename Op>
    static constexpr bool test_masked(span_t<E> args, Op op) noexcept {
        test_masked(dpp::load<A>(args.data()), op);
        return true;
    }

    template <dpp::simd_element_for<A> E, typename Op>
    static constexpr bool test_masked(E arg, Op op) noexcept {
        test_masked(dpp::broadcast<A>(arg), op);
        return true;
    }
};

export template <dpp::simd_abi A>
class binary_predicate {
    template <dpp::simd_element_for<A> E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using mask_t = dpp::basic_mask<E, A>;
    template <dpp::simd_element_for<A> E>
    using span_t = conditional_t<dpp::fixed_width_abi<A>,
        span<E const, abi_traits<E>::size>, span<E const>>;
    template <dpp::simd_element_for<A> E>
    using vec_t = dpp::basic_vector<E, A>;
    template <dpp::simd_element_for<A> E>
    using bitset_t = dpl::bitset<abi_traits<E>::size>;

    template <typename E>
    static constexpr bool all_equal(mask_t<E> lhs, mask_t<E> rhs) noexcept {
        return dpp::to_bitset(lhs) == dpp::to_bitset(rhs);
    }

    template <typename E, typename Op>
    static constexpr void test_masked(
        vec_t<E> const lhs, vec_t<E> const rhs, Op const op) noexcept {
        auto const all_true = dpp::broadcast<E, A>(true);
        auto const all_false = dpp::broadcast<E, A>(false);
        auto const alt_mask = []() {
            if constexpr (dpp::fixed_width_abi<A>) {
                constexpr auto lanes = abi_traits<E>::size();
                return dpp::from_bitset<A, E>(test::repeat_byte<lanes>(0x55u));
            } else {
                return dpp::cmpeq(
                    dpp::bwand(dpp::lane_index<A, E>(), 1), dpp::zero);
            }
        }();
        auto const vop = op(lhs, rhs);

        // merge-masked: active → vop, inactive → src
        {
            assert(all_equal<E>(op(all_true, lhs, rhs), vop));
            assert(all_equal<E>(op(all_false, lhs, rhs), all_false));
            assert(all_equal<E>(
                op(alt_mask, lhs, rhs), dpp::bwand(vop, alt_mask)));
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
                assert(all_equal<E>(op(all_cmask, lhs, rhs), vop));
                assert(all_equal<E>(op(none_cmask, lhs, rhs), all_false));
                assert(all_equal<E>(
                    op(alt_cmask, lhs, rhs), op(alt_mask, lhs, rhs)));
            }
        }
    }

public:
    template <dpp::simd_element_for<A> E, typename Op>
    static constexpr bool test(
        span_t<E> lhs, span_t<E> rhs, Op op, bitset_t<E> expected) noexcept
    requires dpp::fixed_width_abi<A>
    {
        auto const vlhs = dpp::load<A>(lhs.data());
        auto const vrhs = dpp::load<A>(rhs.data());
        auto const vactual = op(vlhs, vrhs);
        assert(dpp::to_bitset(vactual) == expected);
        return true;
    }

    template <dpp::simd_element_for<A> E, typename Op>
    static constexpr bool test(
        E lhs, E rhs, Op op, bitset_t<E> expected) noexcept {
        auto const vlhs = dpp::broadcast<A>(lhs);
        auto const vrhs = dpp::broadcast<A>(rhs);
        auto const vactual = op(vlhs, vrhs);
        assert(dpp::to_bitset(vactual) == expected);
        return true;
    }

    template <dpp::simd_element_for<A> E, typename Op>
    static constexpr bool test_masked(
        span_t<E> lhs, span_t<E> rhs, Op op) noexcept {
        test_masked(dpp::load<A>(lhs.data()), dpp::load<A>(rhs.data()), op);
        return true;
    }

    template <dpp::simd_element_for<A> E, typename Op>
    static constexpr bool test_masked(E lhs, E rhs, Op op) noexcept {
        test_masked(dpp::broadcast<A>(lhs), dpp::broadcast<A>(rhs), op);
        return true;
    }
};
} // namespace support
} // namespace dpl::test
