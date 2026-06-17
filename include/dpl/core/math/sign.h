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

DPL_DEFAULT_NAMESPACE_BEGIN
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
    private math_operation_base<sign_t>,
    private maskable_transform_base<sign_t> {
    using math_operation_base<sign_t>::operator();
    using maskable_transform_base<sign_t>::operator();
};

template <>
struct operation_signature<sign_t> {
    static consteval void operator()(
        simd_vector auto&&, simd_vector auto&&) noexcept {}
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_sign = requires {
    {
        sign(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<basic_vector<simd_element_type_t<L>, A>>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_canonical_msign =
    (!simd_type<S> || same_as<S, cpo_result_t<sign_t, L, R>>) && requires {
        {
            sign(internal::abi<cpo_result_t<sign_t, L, R>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<sign_t, L, R>>;
    };

template <>
struct canonical_impl<sign_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

    template <typename L, typename R>
    using vresult_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using vmask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using vimask_t DPL_NODEBUG = imask_t<cpo_result_t<sign_t, L, R>>;

    template <typename L, typename R, vimask_t<L, R> M>
    using vcmask_t DPL_NODEBUG = cmask_t<cpo_result_t<sign_t, L, R>, M>;

public:
    template <canonical_vector L, common_vector_with<L> R = L>
    requires canonical_vector<R> && unqualified_canonical_sign<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(L lhs, R rhs) noexcept {
        return sign(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_msign<vresult_t<L, R>, vmask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return sign(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, vimask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_msign<vresult_t<L, R>, vcmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, vcmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return sign(internal::abi<common_abi_t<L, R>>, src, cmask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_msign<dx::zero_t, vmask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return sign(internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, vimask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_msign<dx::zero_t, vcmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, vcmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return sign(internal::abi<common_abi_t<L, R>>, zero, cmask, lhs, rhs);
    }
};

template <typename L, typename R = L, typename A = common_abi_t<L, R>>
concept unqualified_extended_sign = requires {
    {
        sign(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_msign =
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<sign_t, L, R>>) &&
    requires {
        {
            sign(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<sign_t, L, R>>;
    };

template <>
struct extended_impl<sign_t> {
private:
    template <typename S>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<S>::size>;

    template <typename S, imask_t<S> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<S>::size, V>;

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename L, typename R>
    using vimask_t DPL_NODEBUG = imask_t<cpo_result_t<sign_t, L, R>>;

    template <typename L, typename R, vimask_t<L, R> M>
    using vcmask_t DPL_NODEBUG = cmask_t<cpo_result_t<sign_t, L, R>, M>;

    template <typename L, typename R>
    using vmask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_sign<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return sign(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_msign<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return sign(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msign<S, cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, cmask_t<S, M> cmask, L&& lhs, R&& rhs) {
        return sign(src, cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        common_mask_with<vmask_t<L, R>> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msign<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return sign(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R, vimask_t<L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msign<dx::zero_t, vcmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, vcmask_t<L, R, M> cmask, L&& lhs, R&& rhs) {
        return sign(
            zero, cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct fallback_impl<sign_t> {
    template <simd_abi A, simd_element_for<A> E>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unsigned_integral<E>) {
            return dx::select(dx::cmpeq(dx::zero, rhs), dx::zero, lhs);
        } else {
            return dx::select(dx::cmpeq(dx::zero, rhs), dx::zero,
                dx::negate(lhs, dx::cmplt(rhs, dx::zero), lhs));
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return dx::bwxor(lhs, dx::bwand(rhs, dx::msb));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sign_t sign{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
