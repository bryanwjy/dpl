// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/bitwise.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void slide_left(...) noexcept = delete;

struct DPL_EMPTY_BASES slide_left_t :
    public algorithm_base<slide_left_t>,
    public maskable_transform_base<slide_left_t> {
    using operation_base<slide_left_t>::operator();
    using maskable_transform_base<slide_left_t>::operator();
};

template <>
struct operation_signature<slide_left_t> {
    template <simd_vector L, equivalent_vector_with<L> R,
        integral_constant_like N>
    static consteval void operator()(L&&, R&&, N) noexcept {}

    template <simd_vector L, broadcastable_to<L> R, integral_constant_like N>
    static consteval void operator()(L&&, R&&, N) noexcept {}

    template <simd_vector R, broadcastable_to<R> L, integral_constant_like N>
    static consteval void operator()(L&&, R&&, N) noexcept {}

    template <simd_vector L, equivalent_vector_with<L> R>
    static consteval void operator()(L&&, R&&, size_t) noexcept {}

    template <simd_vector L, broadcastable_to<L> R>
    static consteval void operator()(L&&, R&&, size_t) noexcept {}

    template <simd_vector R, broadcastable_to<R> L>
    static consteval void operator()(L&&, R&&, size_t) noexcept {}
};

template <>
struct fallback_impl<slide_left_t> {
    template <simd_vector L, equivalent_vector_with<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        L&& lhs, R&& rhs, size_t num) noexcept {
        using E = simd_element_type_t<L>;
        auto const simd_size = simd_abi_traits<L>::size();
        num = num <= simd_size ? num : simd_size;
        auto const low = dx::shift_left(__DPL forward<L>(lhs), num);
        auto const high =
            dx::shift_right(__DPL forward<R>(rhs), simd_size - num);
        return dx::bwor(low, high);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs, size_t count) noexcept {
        return operator()(
            lhs, dx::broadcast<L>(__DPL forward<R>(rhs)), count);
    }

    template <canonical_vector R, broadcastable_to<R> L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs, size_t count) noexcept {
        return operator()(
            dx::broadcast<R>(__DPL forward<L>(lhs)), rhs, count);
    }
};

template <typename L, typename R, typename N = size_t,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_slide_left = requires {
    {
        slide_left(internal::abi<T>, internal::declarg<L>(),
            internal::declarg<R>(), internal::declarg<N>())
    } -> same_as<T>;
};

template <typename S, typename M, typename L, typename R, typename N = size_t,
    typename A = common_abi_t<L, R>>
concept unqualified_canonical_mslide_left =
    cpo_invocable<slide_left_t, L, R, N> &&
    (!simd_type<S> || same_as<S, cpo_result_t<slide_left_t, L, R, N>>) &&
    requires {
        {
            slide_left(internal::abi<A>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>(), internal::declarg<N>())
        } -> same_as<cpo_result_t<slide_left_t, L, R, N>>;
    };

template <>
struct canonical_impl<slide_left_t> {
    template <canonical_vector T>
    requires unqualified_canonical_slide_left<T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T lhs, type_identity_t<T> rhs, size_t count) noexcept {
        return slide_left(internal::abi<T>, lhs, rhs, count);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_slide_left<L, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs, size_t count) noexcept {
        return slide_left(internal::abi<L>, lhs, __DPL forward<R>(rhs), count);
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_slide_left<R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs, size_t count) noexcept {
        return slide_left(internal::abi<R>, __DPL forward<L>(lhs), rhs, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mslide_left<T, simd_mask_type_t<T>, T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T lhs, type_identity_t<T> rhs,
        size_t count) noexcept {
        return slide_left(internal::abi<T>, src, mask, lhs, rhs, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mslide_left<T, launder_cmask_t<T, M>, T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, M cmask, T lhs,
        type_identity_t<T> rhs, size_t count) noexcept {
        return slide_left(internal::abi<T>, src, dx::to_const_mask<T>(cmask),
            lhs, rhs, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mslide_left<dx::zero_t, simd_mask_type_t<T>,
        T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T lhs, type_identity_t<T> rhs, size_t count) noexcept {
        return slide_left(internal::abi<T>, zero, mask, lhs, rhs, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mslide_left<dx::zero_t,
        launder_cmask_t<T, M>, T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T lhs,
        type_identity_t<T> rhs, size_t count) noexcept {
        return slide_left(internal::abi<T>, zero, dx::to_const_mask<T>(cmask),
            lhs, rhs, count);
    }

    ///
    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_slide_left<T, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T lhs, type_identity_t<T> rhs, N count) noexcept {
        return slide_left(internal::abi<T>, lhs, rhs, count);
    }

    template <canonical_vector L, broadcastable_to<L> R,
        integral_constant_like N>
    requires unqualified_canonical_slide_left<L, L, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs, size_t count) noexcept {
        return slide_left(internal::abi<L>, lhs,
            dx::broadcast<R>(__DPL forward<R>(rhs)), count);
    }

    template <canonical_vector R, broadcastable_to<R> L,
        integral_constant_like N>
    requires unqualified_canonical_slide_left<R, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs, size_t count) noexcept {
        return slide_left(internal::abi<R>,
            dx::broadcast<R>(__DPL forward<L>(lhs)), rhs, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mslide_left<T, simd_mask_type_t<T>, T, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T lhs, type_identity_t<T> rhs,
        N count) noexcept {
        return slide_left(internal::abi<T>, src, mask, lhs, rhs, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires unqualified_canonical_mslide_left<T, launder_cmask_t<T, M>, T, T,
        N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, M cmask, T lhs,
        type_identity_t<T> rhs, N count) noexcept {
        return slide_left(internal::abi<T>, src, dx::to_const_mask<T>(cmask),
            lhs, rhs, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mslide_left<dx::zero_t, simd_mask_type_t<T>,
        T, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T lhs, type_identity_t<T> rhs, N count) noexcept {
        return slide_left(internal::abi<T>, zero, mask, lhs, rhs, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires unqualified_canonical_mslide_left<dx::zero_t,
        launder_cmask_t<T, M>, T, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T lhs,
        type_identity_t<T> rhs, N count) noexcept {
        return slide_left(internal::abi<T>, zero, dx::to_const_mask<T>(cmask),
            lhs, rhs, count);
    }
};

template <typename L, typename R, typename N = size_t>
concept unqualified_extended_slide_left = requires {
    {
        slide_left(internal::declarg<L>(), internal::declarg<R>(),
            internal::declarg<N>())
    } -> vector_with_common_abi<
        simd_abi_type_t<conditional_t<simd_vector<L>, L, R>>>;
};

template <typename S, typename M, typename L, typename R, typename N = size_t>
concept unqualified_extended_mslide_left =
    cpo_invocable<slide_left_t, L, R, N> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<slide_left_t, L, R, N>>) &&
    requires {
        {
            slide_left(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>(),
                internal::declarg<N>())
        } -> equivalent_vector_with<cpo_result_t<slide_left_t, L, R, N>>;
    };

template <>
struct extended_impl<slide_left_t> {
public:
    template <simd_vector L, equivalent_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_slide_left<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs, size_t count) {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <extended_vector L, broadcastable_to<L> R>
    requires unqualified_extended_slide_left<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs, size_t count) noexcept {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <extended_vector R, broadcastable_to<R> L>
    requires unqualified_extended_slide_left<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs, size_t count) noexcept {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> L,
        equivalent_vector_with<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mslide_left<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, L&& lhs, R&& rhs, size_t count) {
        return slide_left(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> L,
        equivalent_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mslide_left<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, L&& lhs, R&& rhs, size_t count) {
        return slide_left(src, dx::to_const_mask<S>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector L, equivalent_vector_with<L> R,
        result_mask_for<slide_left_t, L, R, size_t> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mslide_left<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs, size_t count) {
        return slide_left(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs), count);
    }

    template <simd_vector L, equivalent_vector_with<L> R,
        result_cmask_for<slide_left_t, L, R, size_t> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mslide_left<dx::zero_t,
            launder_cmask_t<cpo_result_t<slide_left_t, L, R, size_t>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs, size_t count) {
        return slide_left(zero,
            dx::to_const_mask<cpo_result_t<slide_left_t, L, R, size_t>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    ///
    template <simd_vector L, equivalent_vector_with<L> R,
        integral_constant_like N>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_slide_left<L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs, N count) {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <extended_vector L, broadcastable_to<L> R,
        integral_constant_like N>
    requires unqualified_extended_slide_left<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs, N count) noexcept {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <extended_vector R, broadcastable_to<R> L,
        integral_constant_like N>
    requires unqualified_extended_slide_left<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs, N count) noexcept {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> L,
        equivalent_vector_with<L> R, integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mslide_left<S, M, L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, L&& lhs, R&& rhs, N count) {
        return slide_left(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> L,
        equivalent_vector_with<L> R, integral_constant_like N>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mslide_left<S, launder_cmask_t<S, M>, L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, L&& lhs, R&& rhs, N count) {
        return slide_left(src, dx::to_const_mask<S>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector L, equivalent_vector_with<L> R,
        integral_constant_like N, result_mask_for<slide_left_t, L, R, N> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mslide_left<dx::zero_t, M, L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs, N count) {
        return slide_left(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs), count);
    }

    template <simd_vector L, equivalent_vector_with<L> R,
        integral_constant_like N, result_cmask_for<slide_left_t, L, R, N> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mslide_left<dx::zero_t,
            launder_cmask_t<cpo_result_t<slide_left_t, L, R, N>, M>, L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs, N count) {
        return slide_left(zero,
            dx::to_const_mask<cpo_result_t<slide_left_t, L, R, N>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }
};

template <size_t V>
struct slide_lefti_t {
    template <simd_vector L, equivalent_vector_with<L> R>
    requires cpo_invocable<slide_left_t, L, R, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) noexcept(
        canonical_vector<L> && canonical_vector<R>) {
        return slide_left_t::operator()(
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), imm<V>);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <auto V>
inline constexpr internal::slide_lefti_t<V> slide_lefti{};
inline constexpr internal::slide_left_t slide_left{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
