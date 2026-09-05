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
        auto low = dx::shift_left(__DPL forward<L>(lhs), num);
        auto high = dx::shift_right(__DPL forward<R>(rhs), simd_size - num);
        return dx::bwor(__DPL move(low), __DPL move(high));
    }

    template <simd_vector L, broadcastable_to<L> R>
    requires cpo_invocable<slide_left_t, L, canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<slide_left_t, L, canonical_type_t<L>>
    operator()(L&& lhs, R&& rhs, size_t count) noexcept(canonical_vector<L>) {
        return slide_left_t::operator()( __DPL forward<L>(lhs),
            dx::broadcast<L>(__DPL forward<R>(rhs)), count);
    }

    template <simd_vector R, broadcastable_to<R> L>
    requires cpo_invocable<slide_left_t, L, canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<slide_left_t, R, canonical_type_t<R>>
    operator()(L&& lhs, R&& rhs, size_t count) noexcept(canonical_vector<L>) {
        return slide_left_t::operator()(
            dx::broadcast<R>(__DPL forward<L>(lhs)), __DPL forward<R>(rhs),
            count);
    }
};

template <typename L, typename R = L, typename N = size_t,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_slide_left = requires {
    {
        slide_left(internal::abi<T>, internal::declarg<L>(),
            internal::declarg<R>(), internal::declarg<N>())
    } -> same_as<T>;
};

template <typename S, typename M, typename L, typename R, typename N = size_t>
concept unqualified_canonical_mslide_left =
    cpo_invocable<slide_left_t, L, R, N> && requires {
        {
            slide_left(internal::abi<cpo_result_t<slide_left_t, L, R, N>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>(),
                internal::declarg<N>())
        } -> same_as<cpo_result_t<slide_left_t, L, R, N>>;
    };

template <>
struct canonical_impl<slide_left_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = conditional_t<canonical_vector<L>, L, R>;
    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

public:
    template <canonical_vector L, same_as<L> R>
    requires unqualified_canonical_slide_left<L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R rhs, size_t count) noexcept {
        return slide_left(internal::abi<L>, lhs, rhs, count);
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

    template <unextended_type L, unextended_type R>
    requires unqualified_canonical_mslide_left<result_t<L, R>, mask_t<L, R>, L,
        R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(result_t<L, R> src,
        mask_t<L, R> mask, L&& lhs, R&& rhs, size_t count) noexcept {
        return slide_left(internal::abi<result_t<L, R>>, src, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <unextended_type L, unextended_type R,
        result_cmask_for<L, R, size_t> M>
    requires unqualified_canonical_mslide_left<result_t<L, R>,
        launder_cmask_t<result_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M mask, L&& lhs, R&& rhs, size_t count) noexcept {
        return slide_left(internal::abi<result_t<L, R>>, src,
            dx::to_const_mask<result_t<L, R>>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs), count);
    }

    template <unextended_type L, unextended_type R>
    requires unqualified_canonical_mslide_left<dx::zero_t, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(dx::zero_t zero,
        mask_t<L, R> mask, L&& lhs, R&& rhs, size_t count) noexcept {
        return slide_left(internal::abi<result_t<L, R>>, zero, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <unextended_type L, unextended_type R,
        result_cmask_for<L, R, size_t> M>
    requires unqualified_canonical_mslide_left<dx::zero_t,
        launder_cmask_t<result_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs, size_t count) noexcept {
        return slide_left(internal::abi<result_t<L, R>>, zero,
            dx::to_const_mask<result_t<L, R>>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs), count);
    }

    ///
    template <canonical_vector L, same_as<L> R, integral_constant_like N>
    requires unqualified_canonical_slide_left<L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R rhs, N count) noexcept {
        return slide_left(internal::abi<L>, lhs, rhs, count);
    }

    template <canonical_vector L, broadcastable_to<L> R,
        integral_constant_like N>
    requires unqualified_canonical_slide_left<L, L, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs, N count) noexcept {
        return slide_left(internal::abi<L>, lhs, __DPL forward<R>(rhs), count);
    }

    template <canonical_vector R, broadcastable_to<R> L,
        integral_constant_like N>
    requires unqualified_canonical_slide_left<R, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs, N count) noexcept {
        return slide_left(internal::abi<R>, __DPL forward<L>(lhs), rhs, count);
    }

    template <unextended_type L, unextended_type R, integral_constant_like N>
    requires unqualified_canonical_mslide_left<result_t<L, R>, mask_t<L, R>, L,
        R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(result_t<L, R> src,
        mask_t<L, R> mask, L&& lhs, R&& rhs, N count) noexcept {
        return slide_left(internal::abi<result_t<L, R>>, src, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <unextended_type L, unextended_type R, integral_constant_like N,
        result_cmask_for<L, R, N> M>
    requires unqualified_canonical_mslide_left<result_t<L, R>,
        launder_cmask_t<result_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M mask, L&& lhs, R&& rhs, N count) noexcept {
        return slide_left(internal::abi<result_t<L, R>>, src,
            dx::to_const_mask<result_t<L, R>>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs), count);
    }

    template <unextended_type L, unextended_type R, integral_constant_like N>
    requires unqualified_canonical_mslide_left<dx::zero_t, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(dx::zero_t zero,
        mask_t<L, R> mask, L&& lhs, R&& rhs, N count) noexcept {
        return slide_left(internal::abi<result_t<L, R>>, zero, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <unextended_type L, unextended_type R, integral_constant_like N,
        result_cmask_for<L, R, N> M>
    requires unqualified_canonical_mslide_left<dx::zero_t,
        launder_cmask_t<result_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs, N count) noexcept {
        return slide_left(internal::abi<result_t<L, R>>, zero,
            dx::to_const_mask<result_t<L, R>>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs), count);
    }
};

template <typename L, typename R, typename T, typename N = size_t>
concept unqualified_extended_slide_left = requires {
    {
        slide_left(internal::declarg<L>(), internal::declarg<R>(),
            internal::declarg<N>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename L, typename R, typename N = size_t>
concept unqualified_extended_mslide_left_base =
    cpo_invocable<slide_left_t, L, R, N> && requires {
        {
            slide_left(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>(),
                internal::declarg<N>())
        } -> equivalent_vector_with<cpo_result_t<slide_left_t, L, R, N>>;
    };

template <typename S, typename M, typename L, typename R, typename N = size_t>
concept unqualified_extended_mslide_left =
    unqualified_extended_mslide_left_base<S, M, L, R, N> &&
    equivalent_vector_with<S, cpo_result_t<slide_left_t, L, R, N>>;

template <typename M, typename L, typename R, typename N = size_t>
concept unqualified_extended_zmslide_left =
    unqualified_extended_mslide_left_base<dx::zero_t, M, L, R, N>;

template <>
struct extended_impl<slide_left_t> {
public:
    template <simd_vector L, equivalent_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_slide_left<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs, size_t count) {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <extended_vector L, broadcastable_to<L> R>
    requires unqualified_extended_slide_left<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R&& rhs, size_t count) noexcept {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <extended_vector R, broadcastable_to<R> L>
    requires unqualified_extended_slide_left<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R rhs, size_t count) noexcept {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mslide_left<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, L&& lhs, R&& rhs, size_t count) {
        return slide_left(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mslide_left<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M mask, L&& lhs, R&& rhs, size_t count) {
        return slide_left(src, dx::to_const_mask<S>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <typename L, typename R,
        result_mask_for<slide_left_t, L, R, size_t> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmslide_left<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs, size_t count) {
        return slide_left(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs), count);
    }

    template <typename L, typename R,
        result_cmask_for<slide_left_t, L, R, size_t> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmslide_left<
            launder_cmask_t<cpo_result_t<slide_left_t, L, R, size_t>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs, size_t count) {
        return slide_left(zero,
            dx::to_const_mask<cpo_result_t<slide_left_t, L, R, size_t>>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    ///
    template <simd_vector L, equivalent_vector_with<L> R,
        integral_constant_like N>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_slide_left<L, R, L, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs, N count) {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <extended_vector L, broadcastable_to<L> R,
        integral_constant_like N>
    requires unqualified_extended_slide_left<L, R, L, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R&& rhs, N count) noexcept {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <extended_vector R, broadcastable_to<R> L,
        integral_constant_like N>
    requires unqualified_extended_slide_left<L, R, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R rhs, N count) noexcept {
        return slide_left(__DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mslide_left<S, M, L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, L&& lhs, R&& rhs, N count) {
        return slide_left(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mslide_left<S, launder_cmask_t<S, M>, L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M mask, L&& lhs, R&& rhs, N count) {
        return slide_left(src, dx::to_const_mask<S>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs), count);
    }

    template <typename L, typename R, integral_constant_like N,
        result_mask_for<slide_left_t, L, R, N> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmslide_left<M, L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs, N count) {
        return slide_left(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs), count);
    }

    template <typename L, typename R, integral_constant_like N,
        result_cmask_for<slide_left_t, L, R, N> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmslide_left<
            launder_cmask_t<cpo_result_t<slide_left_t, L, R, N>, M>, L, R, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs, N count) {
        return slide_left(zero,
            dx::to_const_mask<cpo_result_t<slide_left_t, L, R, N>>(mask),
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
