// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/select.h"

#if !DPL_MODULES
#  include "dpl/core/basic/zero.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/type_traits/declval.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename Op, fixed_width_abi A, simd_element_for<A> PassE,
    simd_element_for<A> MaskE, simd_element_for<A>... InEs, typename... InArgs>
requires common_size_with<MaskE, PassE> &&
    regular_invocable<Op, basic_vector<InEs, A>..., InArgs...> &&
    same_as<invoke_result_t<Op, basic_vector<InEs, A>..., InArgs...>,
        basic_vector<PassE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr basic_vector<PassE, A> masked(
    basic_vector<PassE, A> pass, basic_mask<MaskE, A> mask,
    basic_vector<InEs, A>... args, InArgs... others) noexcept {
    constexpr Op operation{};
    return dx::select(mask, operation(args..., others...), pass);
}

template <typename Op, fixed_width_abi A, simd_element_for<A> MaskE,
    simd_element_for<A>... InEs, typename... InArgs>
requires regular_invocable<Op, basic_vector<InEs, A>..., InArgs...> &&
    common_size_with<MaskE,
        typename invoke_result_t<Op, basic_vector<InEs, A>...,
            InArgs...>::value_type>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto masked(basic_mask<MaskE, A> mask,
    basic_vector<InEs, A>... args, InArgs... others) noexcept {
    constexpr Op operation{};
    return dx::select(mask, operation(args..., others...), dx::zero);
}

template <typename Op, fixed_width_abi A, simd_element_for<A> PassE,
    const_mask_for<basic_vector<PassE, A>> M, simd_element_for<A>... InEs,
    typename... InArgs>
requires regular_invocable<Op, basic_vector<InEs, A>..., InArgs...> &&
    same_as<invoke_result_t<Op, basic_vector<InEs, A>..., InArgs...>,
        basic_vector<PassE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr basic_vector<PassE, A> masked(
    basic_vector<PassE, A> pass, M mask, basic_vector<InEs, A>... args,
    InArgs... others) noexcept {
    constexpr Op operation{};
    return dx::select(mask, operation(args..., others...), pass);
}

template <typename Op, fixed_width_abi A, typename M,
    simd_element_for<A>... InEs, typename... InArgs>
requires regular_invocable<Op, basic_vector<InEs, A>..., InArgs...> &&
    const_mask_for<M, invoke_result_t<Op, basic_vector<InEs, A>..., InArgs...>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto masked(
    M mask, basic_vector<InEs, A>... args, InArgs... others) noexcept {
    constexpr Op operation{};
    return dx::select(mask, operation(args..., others...), dx::zero);
}

template <typename Op, typename... Ts>
using operation_result_t DPL_NODEBUG =
    decltype(__DPL declval<Op const>()( __DPL declval<Ts>()...));

template <typename T, typename U, typename A = typename U::abi_type>
using canonical_if_zero_t DPL_NODEBUG = conditional_t<is_same_v<T, zero_t>,
    basic_vector<simd_lane_type_t<U>, A>, T>;

template <typename T, typename U, typename A = typename U::abi_type>
using canonical_or_zero_t DPL_NODEBUG = conditional_t<is_same_v<T, zero_t>, T,
    basic_vector<simd_lane_type_t<U>, A>>;

template <typename A, typename Op, typename S, typename... Args, typename M>
consteval auto to_const_mask(M mask) noexcept {
    using T = operation_result_t<Op, Args...>;
    return dx::to_compatible_const_mask<canonical_if_zero_t<S, T, A>>(mask);
}

template <typename S, typename M, typename... Args>
concept maskable_args =
    simd_vector<S> && simd_mask<M> && (... && simd_vector<Args>) &&
    same_abi_as<typename S::abi_type, typename M::abi_type> &&
    all_common_abi<typename Args::abi_type...> &&
    common_abi_with<common_abi_t<Args...>, typename S::abi_type> &&
    common_abi_with<common_abi_t<Args...>, typename M::abi_type> &&
    same_abi_as<common_abi_t<typename S::abi_type, Args...>,
        common_abi_t<typename M::abi_type, Args...>>;

template <typename M, typename... Args>
concept zmaskable_args = simd_mask<M> && (... && simd_vector<Args>) &&
    all_common_abi<typename Args::abi_type...> &&
    common_abi_with<common_abi_t<Args...>, typename M::abi_type>;

template <typename S, typename... Args>
concept imm_maskable_args = simd_vector<S> && (... && simd_vector<Args>) &&
    all_common_abi<typename Args::abi_type...> &&
    common_abi_with<common_abi_t<Args...>, typename S::abi_type>;

template <typename... Args>
concept imm_zmaskable_args =
    (... && simd_vector<Args>) && all_common_abi<typename Args::abi_type...>;

} // namespace datapar::internal

namespace datapar {

DPL_EXPORT struct masked_operation_t {
    explicit consteval masked_operation_t() noexcept = default;

    template <integral_constant_like T>
    consteval operator T(this masked_operation_t) noexcept {
        return T{};
    }
};

DPL_EXPORT inline constexpr masked_operation_t masked_operation{};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
