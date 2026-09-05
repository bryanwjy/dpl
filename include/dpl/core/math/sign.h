// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/msb.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
/**
 * Returns the negation of the left argument if the right argument is
 * negative.
 *
 * If the element types are unsigned, the left argument will be returned
 * unmodified.
 *
 * For integral elements, if the right argument is 0 the result will also be
 * zero.
 *
 * For floating point elements, the result is unaffected by 0.0 but a
 * -0.0 value on the right will negate the left.
 */
void sign(...) noexcept = delete;

struct DPL_EMPTY_BASES sign_t :
    public math_operation_base<sign_t>,
    public maskable_transform_base<sign_t> {
    using math_operation_base<sign_t>::operator();
    using maskable_transform_base<sign_t>::operator();
};

template <>
struct operation_signature<sign_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<sign_t> : binary_canonical_broadcaster<sign_t> {

    template <canonical_vector L, same_as<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr L DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        using E = simd_element_type_t<L>;
        if constexpr (unsigned_integral<E>) {
            return dx::select(dx::cmpeq(dx::zero, rhs), dx::zero, lhs);
        } else if constexpr (floating_point_like<E>) {
            return dx::bwxor(lhs, dx::bwand(rhs, dx::msb));
        } else {
            return dx::select(dx::cmpeq(dx::zero, rhs), dx::zero,
                dx::negate(lhs, dx::cmplt(rhs, dx::zero), lhs));
        }
    }

    using binary_canonical_broadcaster<sign_t>::operator();
};

template <typename L, typename R, typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_sign = requires {
    {
        sign(internal::abi<T>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<T>;
};

template <typename S, typename M, typename L, typename R, typename T>
concept unqualified_canonical_msign_base =
    cpo_invocable<sign_t, L, R> && requires {
        {
            sign(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<T>;
    };

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_msign =
    unqualified_canonical_msign_base<T, M, L, R, T>;

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_zmsign =
    unqualified_canonical_msign_base<dx::zero_t, M, L, R, T>;

template <>
struct canonical_impl<sign_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = cpo_result_t<sign_t, L, R>;
    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

public:
    template <canonical_vector L, same_as<L> R>
    requires canonical_vector<R> && unqualified_canonical_sign<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<L, R> operator()(
        L lhs, R rhs) noexcept {
        return sign(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_sign<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs) noexcept {
        return sign(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_sign<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs) noexcept {
        return sign(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <unextended_type L, unextended_terminal_of<sign_t, L> R>
    requires unqualified_canonical_msign<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return sign(internal::abi<result_t<L, R>>, src, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_msign<launder_cmask_t<result_t<L, R>, M>, L,
        R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return sign(internal::abi<T>, src, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_terminal_of<sign_t, L> R>
    requires unqualified_canonical_zmsign<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return sign(internal::abi<result_t<L, R>>, zero, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_zmsign<launder_cmask_t<result_t<L, R>, M>, L,
        R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return sign(internal::abi<T>, zero, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <typename L, typename R, typename T>
concept unqualified_extended_sign = requires {
    {
        sign(internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_msign_base =
    cpo_invocable<sign_t, L, R> && requires {
        {
            sign(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<sign_t, L, R>>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_msign =
    equivalent_vector_with<S, cpo_result_t<sign_t, L, R>> &&
    unqualified_extended_msign_base<S, M, L, R>;

template <typename M, typename L, typename R>
concept unqualified_extended_zmsign =
    unqualified_extended_msign_base<dx::zero_t, M, L, R>;

template <>
struct extended_impl<sign_t> {
public:
    template <simd_vector L, equivalent_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_sign<L, R, common_canonical_simd_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return sign(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector L, broadcastable_to<L> R>
    requires unqualified_extended_sign<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return sign(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector R, broadcastable_to<R> L>
    requires unqualified_extended_sign<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return sign(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_msign<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return sign(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msign<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, L&& lhs, R&& rhs) {
        return sign(src, dx::to_const_mask<S>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_mask_for<sign_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmsign<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return sign(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_cmask_for<sign_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmsign<
            launder_cmask_t<cpo_result_t<sign_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) {
        return sign(zero, dx::to_const_mask<cpo_result_t<sign_t, L, R>>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::sign_t sign{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
