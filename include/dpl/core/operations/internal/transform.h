// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/std/type_traits/is_invocable.h"
#  include "dpl/std/utility/bitset.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename F, typename R, typename... Args>
concept value_invocable_r =
    (canonical_simd_type<R> && ... && canonical_simd_type<Args>) &&
    (... && (simd_abi_traits<R>::size == simd_abi_traits<Args>::size)) &&
    is_invocable_r_v<typename R::value_type, F, typename Args::value_type...>;

template <typename R, typename F, typename... Args, size_t... Is>
consteval bool invocable(index_sequence<Is...>) noexcept {
    return (... &&
        is_invocable_r_v<typename R::value_type, F, immediate<Is>,
            typename Args::value_type...>);
}

template <typename F, typename R, typename... Args>
concept ivalue_invocable_r =
    (canonical_simd_type<R> && ... && canonical_simd_type<Args>) &&
    (... && (simd_abi_traits<R>::size == simd_abi_traits<Args>::size)) &&
    internal::invocable<R, F, Args...>(iota_sequence<R>);

template <canonical_simd_type Result, canonical_simd_type... Ts,
    ivalue_invocable_r<Result, Ts...> Op>
requires (... && same_abi_as<simd_abi_type_t<Result>, simd_abi_type_t<Ts>>) &&
    (fixed_width_simd_type<Result> && ... && fixed_width_simd_type<Ts>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, FLATTEN, NODISCARD)
constexpr Result itransform(Op func, Ts... args) noexcept {

    constexpr auto single = []<size_t I>(
                                immediate<I> idx, Ts... args, Op func) {
        return __DPL invoke_r<typename Result::value_type>(
            func, idx, args[idx]...);
    };

    return [single]<size_t... Is>(Ts... args, Op func, index_sequence<Is...>) {
        if constexpr (simd_mask<Result>) {
            return Result{bitset(single(imm<Is>, args..., func)...)};
        } else {
            return Result{single(imm<Is>, args..., func)...};
        }
    }(args..., func, iota_sequence<Result>);
}

template <canonical_simd_type Result, canonical_simd_type... Ts,
    value_invocable_r<Result, Ts...> Op>
requires (... && same_abi_as<simd_abi_type_t<Result>, simd_abi_type_t<Ts>>) &&
    (fixed_width_simd_type<Result> && ... && fixed_width_simd_type<Ts>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, FLATTEN, NODISCARD)
constexpr Result transform(Op func, Ts... args) noexcept {
    return internal::itransform<Result>(
        [func](auto, typename Ts::value_type... args) {
            return __DPL invoke_r<typename Result::value_type>(func, args...);
        },
        args...);
}

} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END
