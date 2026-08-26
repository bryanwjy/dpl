// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/add.h"
#include "dpl/core/operations/arithmetic/negate.h"
#include "dpl/core/operations/bitwise/bwand.h"
#include "dpl/core/operations/compare/cmpeq.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/lane_index.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/dispatch/broadcastable/binary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_mask_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
/**
 * @brief Performs a fused SIMD add-sub operation using a fixed lane-parity sign
 * mask.
 *
 * Computes a packed addition where the second operand is sign-modified on a
 * per-lane basis according to lane index parity.
 *
 * Formally, for lane index i:
 *
 * CODE_BLOCK_BEGIN
 * result[i] = a[i] + s[i] * b[i]
 * CODE_BLOCK_END
 *
 * where the sign pattern is fixed as:
 *
 * CODE_BLOCK_BEGIN
 * s[i] = (i % 2 == 0) ? -1 : +1
 * CODE_BLOCK_END
 *
 * This yields the following lane-wise behavior:
 *
 * CODE_BLOCK_BEGIN
 * [a0 - b0, a1 + b1, a2 - b2, a3 + b3, ...]
 * CODE_BLOCK_END
 *
 * This operation is NOT a sequence of alternating addition and subtraction
 * operations. It is a single SIMD addition with a compile-time sign mask
 * applied to the second operand.
 *
 * @note This definition is intentionally fixed and does not depend on operand
 * evaluation order.
 *
 * @note This convention differs from some SIMD ISA definitions where the
 * "addsub" family uses the opposite lane-parity sign pattern:
 *
 *       CODE_BLOCK_BEGIN
 *       [a0 + b0, a1 - b1, a2 + b2, a3 - b3, ...]
 *       CODE_BLOCK_END
 *
 *       In those ISAs, the opposite polarity is exposed either via a separate
 * instruction (e.g. subadd) or as a distinct encoding choice.
 *
 *       DPL standardizes a single fixed convention to avoid ambiguity in
 * cross-ABI behavior, and to align with reading-order lane parity (even = first
 * lane, odd = second lane).
 *
 * @param a First SIMD operand.
 * @param b Second SIMD operand.
 * @return SIMD value containing the fused add-sub result.
 *
 * @warning This operation is layout-sensitive: lane ordering directly
 * determines the sign mask.
 */
void addsub(...) noexcept = delete;

struct DPL_EMPTY_BASES addsub_t :
    public arithmetic_base<addsub_t>,
    public maskable_transform_base<addsub_t>,
    public binary_broadcastable_operation<addsub_t> {
    using operation_base<addsub_t>::operator();
    using maskable_transform_base<addsub_t>::operator();
    using binary_broadcastable_operation<addsub_t>::operator();
};

template <>
struct operation_signature<addsub_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<addsub_t> : binary_broadcasting_fallback<addsub_t> {

    template <simd_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto make_opmask() noexcept {
        if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
            return []<size_t... Is>(index_sequence<Is...>) {
                return cmask_v<__DPL bitset<sizeof...(Is)>(
                    ((Is & 1) == 1)...)>;
            }(iota_sequence<T>);
        } else {
            auto const idx = dx::lane_index<signed_canonical_vector_t<T>>();
            return dx::cmpeq(dx::bwand(idx, dx::one), dx::one);
        }
    }

    template <canonical_vector LT, common_vector_with<LT> RT>
    requires canonical_vector<RT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(LT lhs, RT rhs) noexcept {
        auto const opmask = make_opmask<RT>();
        return dx::add(lhs, dx::negate(rhs, opmask, rhs));
    }

    template <typename L, typename R>
    using result_t DPL_NODEBUG =
        cpo_result_t<add_t, L, cpo_result_t<negate_t, R>>;

    template <canonical_vector LT, common_vector_with<LT> RT>
    requires canonical_vector<RT> &&
        cpo_invocable<add_t, LT, cpo_result_t<negate_t, RT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(result_t<LT, RT> src,
        simd_mask_type_t<result_t<LT, RT>> mask, LT lhs, RT rhs) noexcept {
        auto const opmask = make_opmask<RT>();
        return dx::add(src, mask, lhs, dx::negate(rhs, opmask, rhs));
    }

    template <typename M, canonical_vector LT, common_vector_with<LT> RT>
    requires canonical_vector<RT> &&
        cpo_invocable<add_t, LT, cpo_result_t<negate_t, RT>> &&
        const_mask_for<M, result_t<LT, RT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        result_t<LT, RT> src, M mask, LT lhs, RT rhs) {
        auto const opmask = make_opmask<RT>();
        return dx::add(src, mask, lhs, dx::negate(rhs, opmask, rhs));
    }

    template <canonical_vector LT, common_vector_with<LT> RT>
    requires canonical_vector<RT> &&
        cpo_invocable<add_t, LT, cpo_result_t<negate_t, RT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(dx::zero_t zero,
        simd_mask_type_t<result_t<LT, RT>> mask, LT lhs, RT rhs) noexcept {
        auto const opmask = make_opmask<RT>();
        return dx::add(zero, mask, lhs, dx::negate(rhs, opmask, rhs));
    }

    template <typename M, canonical_vector LT, common_vector_with<LT> RT>
    requires canonical_vector<RT> &&
        cpo_invocable<add_t, LT, cpo_result_t<negate_t, RT>> &&
        const_mask_for<M, cpo_result_t<add_t, LT, cpo_result_t<negate_t, RT>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, LT lhs, RT rhs) noexcept {
        auto const opmask = make_opmask<RT>();
        return dx::add(zero, mask, lhs, dx::negate(rhs, opmask, rhs));
    }

    using binary_broadcasting_fallback<addsub_t>::operator();
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_addsub = requires {
    {
        addsub(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> canonical_vector;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R>>
concept unqualified_canonical_maddsub = cpo_invocable<addsub_t, L, R> &&
    (!simd_type<S> || same_as<S, cpo_result_t<addsub_t, L, R>>) && requires {
        {
            addsub(internal::abi<A>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<cpo_result_t<addsub_t, L, R>>;
    };

template <>
struct canonical_impl<addsub_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && unqualified_canonical_addsub<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return addsub(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_addsub<L, R, simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs) noexcept {
        return addsub(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_addsub<L, R, simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs) noexcept {
        return addsub(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_maddsub<result_t<L, R>, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return addsub(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<result_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_maddsub<result_t<L, R>,
            launder_cmask_t<result_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M cmask, L lhs, R rhs) noexcept {
        return addsub(internal::abi<common_abi_t<L, R>>, src,
            dx::to_const_mask<result_t<L, R>>(cmask), lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_maddsub<dx::zero_t, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return addsub(internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<result_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_maddsub<dx::zero_t,
            launder_cmask_t<result_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M cmask, L lhs, R rhs) noexcept {
        return addsub(internal::abi<common_abi_t<L, R>>, zero,
            dx::to_const_mask<result_t<L, R>>(cmask), lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_addsub = requires {
    {
        addsub(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_maddsub = cpo_invocable<addsub_t, L, R> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<addsub_t, L, R>>) &&
    requires {
        {
            addsub(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<addsub_t, L, R>>;
    };

template <>
struct extended_impl<addsub_t> {
public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_addsub<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return addsub(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R>
    requires (extended_vector<L> &&
                 broadcastable_to<R, result_or_decayed_t<L>> &&
                 unqualified_extended_addsub<L, R, simd_abi_type_t<L>>) ||
        (extended_vector<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
            unqualified_extended_addsub<L, R, simd_abi_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return addsub(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_maddsub<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return addsub(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <fixed_width_vector S, const_mask_for<S> M,
        common_vector_with<S> L, common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_maddsub<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, L&& lhs, R&& rhs) {
        return addsub(src, dx::to_const_mask<S>(cmask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        result_mask_for<addsub_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_maddsub<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return addsub(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        result_cmask_for<addsub_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_maddsub<dx::zero_t,
            launder_cmask_t<cpo_result_t<addsub_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs) {
        return addsub(zero,
            dx::to_const_mask<cpo_result_t<addsub_t, L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::addsub_t addsub{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
