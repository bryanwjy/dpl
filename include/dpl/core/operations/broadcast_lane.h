// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/permute.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/type_traits/sequence.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void broadcast_lane(...) noexcept = delete;

struct DPL_EMPTY_BASES broadcast_lane_t :
    public primitive_operation_base<broadcast_lane_t>,
    public maskable_transform_base<broadcast_lane_t> {
    using operation_base<broadcast_lane_t>::operator();
    using maskable_transform_base<broadcast_lane_t>::operator();
};

template <>
struct operation_signature<broadcast_lane_t> {
    template <simd_vector L, integral_constant_like R>
    static consteval void operator()(L&&, R) noexcept {}
    template <simd_vector L>
    static consteval void operator()(L&&, size_t) noexcept {}
};

template <>
struct fallback_impl<broadcast_lane_t> {
private:
    template <simd_vector T, integral_constant_like I>
    static consteval auto broadcast_sequence(I) noexcept {
        using L = remove_cvref_t<T>;
        return [&]<size_t... Is>(index_sequence<Is...>) {
            return __DPL index_sequence<(
                (Is == Is) ? I::value : I::value)...>{};
        }(iota_sequence<L>);
    }

    template <typename T>
    using index_vector DPL_NODEBUG =
        rebind_simd_t<T, signed_representation_t<simd_element_type_t<T>>>;

public:
    template <simd_vector L, integral_constant_like R>
    requires fixed_width_abi<simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static constexpr auto
    operator()(L val, R idx) noexcept
    requires cpo_invocable<permute_t, L, decltype(broadcast_sequence<L>(idx))>
    {
        constexpr auto seq = broadcast_sequence<L>(idx);
        static_assert(R::value >= 0 && R::value < seq.size());
        return dx::permute(__DPL forward<L>(val), seq);
    }

    template <simd_vector L>
    requires cpo_invocable<broadcast_t<index_vector<L>>,
                 simd_element_type_t<index_vector<L>>> &&
        cpo_invocable<permute_t, L, index_vector<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        L val, size_t idx) noexcept {
        using index_t = simd_element_type_t<index_vector<L>>;
        return dx::permute( __DPL forward<L>(val),
            dx::broadcast<index_vector<L>>(static_cast<index_t>(idx)));
    }
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_canonical_mbroadcast_lane =
    (!simd_type<S> || same_as<S, cpo_result_t<broadcast_lane_t, T, N>>) &&
    requires {
        {
            broadcast_lane(internal::abi<cpo_result_t<broadcast_lane_t, T, N>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> same_as<cpo_result_t<broadcast_lane_t, T, N>>;
    };

template <>
struct canonical_impl<broadcast_lane_t> {
public:
    template <canonical_vector L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L val, size_t idx) noexcept
    requires requires { broadcast_lane(internal::abi<L>, val, idx); }
    {
        return broadcast_lane(internal::abi<L>, val, idx);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbroadcast_lane<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, size_t idx) noexcept {
        return broadcast_lane(internal::abi<T>, src, mask, val, idx);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mbroadcast_lane<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, size_t idx) noexcept {
        return broadcast_lane(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, idx);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbroadcast_lane<dx::zero_t,
        simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, size_t idx) noexcept {
        return broadcast_lane(internal::abi<T>, zero, mask, val, idx);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mbroadcast_lane<dx::zero_t, launder_cmask_t<T, M>,
            T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, size_t idx) noexcept {
        return broadcast_lane(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, idx);
    }

    ///
    template <canonical_vector T, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, N idx) noexcept
    requires requires { broadcast_lane(internal::abi<T>, val, idx); }
    {
        return broadcast_lane(internal::abi<T>, val, idx);
    }

    template <canonical_vector T, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mbroadcast_lane<T, simd_mask_type_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, N idx) noexcept {
        return broadcast_lane(internal::abi<T>, src, mask, val, idx);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mbroadcast_lane<T, launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, N idx) noexcept {
        return broadcast_lane(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, idx);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mbroadcast_lane<dx::zero_t,
        simd_mask_type_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, N idx) noexcept {
        return broadcast_lane(internal::abi<T>, zero, mask, val, idx);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mbroadcast_lane<dx::zero_t, launder_cmask_t<T, M>,
            T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, N idx) noexcept {
        return broadcast_lane(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, idx);
    }
};

template <typename T, typename N = size_t, typename A = simd_abi_type_t<T>>
concept unqualified_extended_broadcast_lane = requires {
    {
        broadcast_lane(internal::declarg<T>(), internal::declarg<N>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_extended_mbroadcast_lane =
    cpo_invocable<cpo_result_t<broadcast_lane_t, T, N>> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<broadcast_lane_t, T, N>>) &&
    requires {
        {
            broadcast_lane(internal::abi<cpo_result_t<broadcast_lane_t, T, N>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> equivalent_vector_with<cpo_result_t<broadcast_lane_t, T, N>>;
    };

template <>
struct extended_impl<broadcast_lane_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_broadcast_lane<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, size_t idx) {
        return broadcast_lane(__DPL forward<T>(val), idx);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbroadcast_lane<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, size_t idx) {
        return broadcast_lane( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), idx);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbroadcast_lane<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, size_t idx) {
        return broadcast_lane( __DPL forward<S>(src),
            dx::to_const_mask<S>(cmask), __DPL forward<T>(val), idx);
    }

    template <simd_vector T, result_mask_for<broadcast_lane_t, T, size_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbroadcast_lane<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, size_t idx) {
        return broadcast_lane(zero, mask, __DPL forward<T>(val), idx);
    }

    template <extended_vector T,
        result_cmask_for<broadcast_lane_t, T, size_t> M>
    requires unqualified_extended_mbroadcast_lane<dx::zero_t,
        launder_cmask_t<cpo_result_t<broadcast_lane_t, T, size_t>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, size_t idx) {
        return broadcast_lane(zero,
            dx::to_const_mask<cpo_result_t<broadcast_lane_t, T, size_t>>(cmask),
            __DPL forward<T>(val), idx);
    }

    ///
    template <extended_vector T, integral_constant_like N>
    requires unqualified_extended_broadcast_lane<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, N idx) {
        return broadcast_lane(__DPL forward<T>(val), idx);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbroadcast_lane<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, N idx) {
        return broadcast_lane( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), idx);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbroadcast_lane<S, launder_cmask_t<S, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, N idx) {
        return broadcast_lane( __DPL forward<S>(src),
            dx::to_const_mask<S>(cmask), __DPL forward<T>(val), idx);
    }

    template <simd_vector T, integral_constant_like N,
        result_mask_for<broadcast_lane_t, T, N> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbroadcast_lane<dx::zero_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, N idx) {
        return broadcast_lane(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), idx);
    }

    template <extended_vector T, integral_constant_like N,
        result_cmask_for<broadcast_lane_t, T, N> M>
    requires unqualified_extended_mbroadcast_lane<dx::zero_t,
        launder_cmask_t<cpo_result_t<broadcast_lane_t, T, N>, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val, N idx) {
        return broadcast_lane(zero,
            dx::to_const_mask<cpo_result_t<broadcast_lane_t, T, N>>(cmask),
            __DPL forward<T>(val), idx);
    }
};

template <size_t I>
struct broadcast_lanei_t {
    template <typename... Ts>
    requires cpo_invocable<broadcast_lane_t, Ts..., size_constant<I>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Ts&&... args) noexcept(
        (... && (!simd_type<Ts> || canonical_simd_type<Ts>))) {
        constexpr size_constant<I> idx;
        return broadcast_lane_t::operator()(__DPL forward<Ts>(args)..., idx);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <size_t I>
inline constexpr internal::broadcast_lanei_t<I> broadcast_lanei{};
inline constexpr internal::broadcast_lane_t broadcast_lane{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
