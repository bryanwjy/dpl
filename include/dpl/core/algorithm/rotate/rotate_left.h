// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/operations/bit/rotl.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void rotate_left(...) noexcept = delete;

struct DPL_EMPTY_BASES rotate_left_t :
    private algorithm_base<rotate_left_t>,
    private maskable_transform_base<rotate_left_t> {
    using operation_base<rotate_left_t>::operator();
    using maskable_transform_base<rotate_left_t>::operator();
};
template <>
struct operation_signature<rotate_left_t> {
    template <simd_vector T, integral_constant_like N>
    static consteval void operator()(T&&, N) noexcept {}
    template <simd_vector T>
    static consteval void operator()(T&&, size_t) noexcept {}
};

template <>
struct fallback_impl<rotate_left_t> {
    template <simd_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, size_t lanes) noexcept {
        lanes %= simd_abi_traits<T>::size();
        return dx::slide_left(val, val, lanes);
    }

    template <fixed_width_vector T, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(T&& val, N lanes) noexcept {
        constexpr auto V = N::value % simd_abi_traits<T>::size();
        return dx::slide_left(val, val, imm<V>);
    }

    template <simd_mask T>
    requires cpo_invocable<rotl_t, T, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, size_t num) noexcept {
        return dx::rotl(__DPL forward<T>(val), num);
    }

    template <simd_mask T, integral_constant_like N>
    requires cpo_invocable<rotl_t, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& val, N num) noexcept {
        return dx::rotl(__DPL forward<T>(val), num);
    }
};

template <typename T, typename N = size_t>
concept unqualified_canonical_rotate_left = requires {
    {
        rotate_left(
            internal::abi<T>, internal::declarg<T>(), internal::declarg<N>())
    } -> same_as<T>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_canonical_mrotate_left =
    cpo_invocable<rotate_left_t, T, N> &&
    (!simd_type<S> || same_as<S, cpo_result_t<rotate_left_t, T, N>>) &&
    requires {
        {
            rotate_left(internal::abi<conditional_t<simd_type<S>, S, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> same_as<cpo_result_t<rotate_left_t, T, N>>;
    };

template <>
struct canonical_impl<rotate_left_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_rotate_left<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, size_t count) noexcept {
        return rotate_left(internal::abi<T>, val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrotate_left<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, size_t count) noexcept {
        return rotate_left(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrotate_left<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        type_identity_t<T> src, M cmask, T val, size_t count) noexcept {
        return rotate_left(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrotate_left<dx::zero_t, simd_mask_type_t<T>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T val, size_t count) noexcept {
        return rotate_left(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrotate_left<dx::zero_t,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T val, size_t count) noexcept {
        return rotate_left(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }

    ///
    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_rotate_left<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N count) noexcept {
        return rotate_left(internal::abi<T>, val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mrotate_left<T, simd_mask_type_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, N count) noexcept {
        return rotate_left(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mrotate_left<T, launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        type_identity_t<T> src, M cmask, T val, N count) noexcept {
        return rotate_left(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mrotate_left<dx::zero_t, simd_mask_type_t<T>,
        T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, N count) noexcept {
        return rotate_left(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires unqualified_canonical_mrotate_left<dx::zero_t,
        launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T val, N count) noexcept {
        return rotate_left(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }
};

template <typename T, typename N = size_t>
concept unqualified_extended_rotate_left = requires {
    {
        rotate_left(internal::declarg<T>(), internal::declarg<N>())
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_extended_mrotate_left =
    cpo_invocable<rotate_left_t, T, N> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<rotate_left_t, T, N>>) &&
    requires {
        {
            rotate_left(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> equivalent_vector_with<cpo_result_t<rotate_left_t, T, N>>;
    };

template <>
struct extended_impl<rotate_left_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_rotate_left<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, size_t count) {
        return rotate_left(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotate_left<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, size_t count) {
        return rotate_left( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrotate_left<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, size_t count) {
        return rotate_left( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector T, result_mask_for<rotate_left_t, T, size_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotate_left<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, T&& val, size_t count) {
        return rotate_left(zero, mask, __DPL forward<T>(val), count);
    }

    template <extended_vector T, result_cmask_for<rotate_left_t, T, size_t> M>
    requires unqualified_extended_mrotate_left<dx::zero_t,
        launder_cmask_t<cpo_result_t<rotate_left_t, T, size_t>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, size_t count) {
        return rotate_left(zero,
            dx::to_const_mask<cpo_result_t<rotate_left_t, T, size_t>>(cmask),
            __DPL forward<T>(val), count);
    }

    ///
    template <extended_vector T, integral_constant_like N>
    requires unqualified_extended_rotate_left<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, N count) {
        return rotate_left(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotate_left<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, N count) {
        return rotate_left( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrotate_left<S, launder_cmask_t<S, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, N count) {
        return rotate_left( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector T, integral_constant_like N,
        result_mask_for<rotate_left_t, T, N> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotate_left<dx::zero_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, N count) {
        return rotate_left(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), count);
    }

    template <extended_vector T, integral_constant_like N,
        result_cmask_for<rotate_left_t, T, N> M>
    requires unqualified_extended_mrotate_left<dx::zero_t,
        launder_cmask_t<cpo_result_t<rotate_left_t, T, N>, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, N count) {
        return rotate_left(zero,
            dx::to_const_mask<cpo_result_t<rotate_left_t, T, N>>(cmask),
            __DPL forward<T>(val), count);
    }
};

template <size_t V>
struct rotate_lefti_t {
public:
    template <typename... Args>
    requires cpo_invocable<rotate_left_t, Args..., immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Args&&... args) noexcept(
        (... && (!simd_type<Args> || canonical_simd_type<Args>))) {
        return rotate_left_t::operator()(__DPL forward<Args>(args)..., imm<V>);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::rotate_left_t rotate_left{};
template <size_t V>
inline constexpr internal::rotate_lefti_t<V> rotate_lefti{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
