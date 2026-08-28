// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/lookup.h"

#if !DPL_MODULES
#  include "dpl/core/basic/lane_index.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/operations/bitwise/bwshift_left.h"
#  include "dpl/core/operations/bitwise/bwshift_right.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void shift_left(...) noexcept = delete;

struct DPL_EMPTY_BASES shift_left_t :
    public algorithm_base<shift_left_t>,
    public maskable_transform_base<shift_left_t> {
    using operation_base<shift_left_t>::operator();
    using maskable_transform_base<shift_left_t>::operator();
};

template <>
struct operation_signature<shift_left_t> {
    template <simd_type T, integral_constant_like N>
    static consteval void operator()(T&&, N) noexcept {}
    template <simd_type T>
    static consteval void operator()(T&&, size_t) noexcept {}
};

template <>
struct fallback_impl<shift_left_t> {
    template <simd_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, size_t num) noexcept {
        using traits = simd_abi_traits<remove_cvref_t<T>>;
        using vidx_t = signed_canonical_vector_t<T>;
        using idx_t = simd_element_type_t<vidx_t>;
        auto const idx =
            dx::add(dx::lane_index<vidx_t>(), static_cast<idx_t>(num));
        return dx::lookup(dx::zero, __DPL forward<T>(val), idx);
    }

    template <simd_mask T>
    requires cpo_invocable<bwshift_left_t, T, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, size_t num) noexcept {
        return dx::bwshift_left(__DPL forward<T>(val), num);
    }

    template <simd_mask T, integral_constant_like N>
    requires cpo_invocable<bwshift_left_t, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& val, N num) noexcept {
        return dx::bwshift_left(__DPL forward<T>(val), num);
    }
};

template <typename T, typename N = size_t>
concept unqualified_canonical_shift_left = requires {
    {
        shift_left(
            internal::abi<T>, internal::declarg<T>(), internal::declarg<N>())
    } -> same_as<T>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_canonical_mshift_left = cpo_invocable<shift_left_t, T, N> &&
    (!simd_type<S> || same_as<S, cpo_result_t<shift_left_t, T, N>>) &&
    requires {
        {
            shift_left(internal::abi<conditional_t<simd_type<S>, S, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> same_as<cpo_result_t<shift_left_t, T, N>>;
    };

template <>
struct canonical_impl<shift_left_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_shift_left<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, size_t count) noexcept {
        return shift_left(internal::abi<T>, val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mshift_left<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, size_t count) noexcept {
        return shift_left(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mshift_left<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        type_identity_t<T> src, M cmask, T val, size_t count) noexcept {
        return shift_left(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mshift_left<dx::zero_t, simd_mask_type_t<T>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T val, size_t count) noexcept {
        return shift_left(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mshift_left<dx::zero_t,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T val, size_t count) noexcept {
        return shift_left(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }

    ///
    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_shift_left<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N count) noexcept {
        return shift_left(internal::abi<T>, val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mshift_left<T, simd_mask_type_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, N count) noexcept {
        return shift_left(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mshift_left<T, launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        type_identity_t<T> src, M cmask, T val, N count) noexcept {
        return shift_left(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mshift_left<dx::zero_t, simd_mask_type_t<T>,
        T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, N count) noexcept {
        return shift_left(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires unqualified_canonical_mshift_left<dx::zero_t,
        launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T val, N count) noexcept {
        return shift_left(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }
};

template <typename T, typename N = size_t>
concept unqualified_extended_shift_left = requires {
    {
        shift_left(internal::declarg<T>(), internal::declarg<N>())
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_extended_mshift_left = cpo_invocable<shift_left_t, T, N> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<shift_left_t, T, N>>) &&
    requires {
        {
            shift_left(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> equivalent_vector_with<cpo_result_t<shift_left_t, T, N>>;
    };

template <>
struct extended_impl<shift_left_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_shift_left<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, size_t count) {
        return shift_left(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mshift_left<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, size_t count) {
        return shift_left( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mshift_left<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, size_t count) {
        return shift_left( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector T, result_mask_for<shift_left_t, T, size_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mshift_left<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, T&& val, size_t count) {
        return shift_left(zero, mask, __DPL forward<T>(val), count);
    }

    template <extended_vector T, result_cmask_for<shift_left_t, T, size_t> M>
    requires unqualified_extended_mshift_left<dx::zero_t,
        launder_cmask_t<cpo_result_t<shift_left_t, T, size_t>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, size_t count) {
        return shift_left(zero,
            dx::to_const_mask<cpo_result_t<shift_left_t, T, size_t>>(cmask),
            __DPL forward<T>(val), count);
    }

    ///
    template <extended_vector T, integral_constant_like N>
    requires unqualified_extended_shift_left<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, N count) {
        return shift_left(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mshift_left<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, N count) {
        return shift_left( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mshift_left<S, launder_cmask_t<S, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, N count) {
        return shift_left( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector T, integral_constant_like N,
        result_mask_for<shift_left_t, T, N> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mshift_left<dx::zero_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, N count) {
        return shift_left(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), count);
    }

    template <extended_vector T, integral_constant_like N,
        result_cmask_for<shift_left_t, T, N> M>
    requires unqualified_extended_mshift_left<dx::zero_t,
        launder_cmask_t<cpo_result_t<shift_left_t, T, N>, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, N count) {
        return shift_left(zero,
            dx::to_const_mask<cpo_result_t<shift_left_t, T, N>>(cmask),
            __DPL forward<T>(val), count);
    }
};

template <size_t V>
struct shift_lefti_t {
public:
    template <simd_type T>
    requires cpo_invocable<shift_left_t, T, immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) noexcept(canonical_simd_type<T>) {
        return shift_left_t::operator()(__DPL forward<T>(val), imm<V>);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::shift_left_t shift_left{};
template <size_t V>
inline constexpr internal::shift_lefti_t<V> shift_lefti{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
