// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/type_traits/details/declarg.h"
#  include "dpl/core/type_traits/simd_value_type.h"
#  include "dpl/std/type_traits/is_invocable.h"
#  include "dpl/std/type_traits/sequence.h"
#  include "dpl/std/utility/bitset.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename F, typename R, typename... Args>
concept value_invocable_r =
    (canonical_simd_type<R> && ... && canonical_simd_type<Args>) &&
    (... && (simd_abi_traits<R>::size == simd_abi_traits<Args>::size)) &&
    requires(F const func, Args&&... args) {
        {
            func(internal::declarg<simd_value_type_t<Args>>()...)
        } -> core_convertible_to<simd_value_type_t<R>>;
    };

template <typename R, typename F, typename... Args, size_t... Is>
constexpr bool value_invocable(index_sequence<Is...>) noexcept {
    return (... &&
        is_invocable_r_v<typename R::value_type, F, immediate<Is>,
            typename Args::value_type...>);
}

template <typename F, typename R, typename... Args>
concept ivalue_invocable_r =
    (canonical_simd_type<R> && ... && canonical_simd_type<Args>) &&
    (... && (simd_abi_traits<R>::size == simd_abi_traits<Args>::size)) &&
    internal::value_invocable<R, F, Args...>(iota_sequence<R>);

template <canonical_simd_type Result, canonical_simd_type... Ts,
    ivalue_invocable_r<Result, Ts...> Op>
requires (... && same_abi_as<simd_abi_type_t<Result>, simd_abi_type_t<Ts>>) &&
    (fixed_width_simd_type<Result> && ... && fixed_width_simd_type<Ts>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, FLATTEN, NODISCARD)
constexpr Result itransform(Op func, Ts... args) noexcept {

    constexpr auto single = []<size_t I>(immediate<I> idx, Op func,
                                Ts... args) -> simd_value_type_t<Result> {
        return func(idx, args[idx]...);
    };

    return [single]<size_t... Is>(Ts... args, Op func, index_sequence<Is...>) {
        if constexpr (simd_mask<Result>) {
            return Result{bitset(single(imm<Is>, func, args...)...)};
        } else {
            return Result{single(imm<Is>, func, args...)...};
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
        [&func](auto, typename Ts::value_type... args) noexcept
            -> simd_value_type_t<Result> { return func(args...); },
        args...);
}

} // namespace datapar::internal
__DPL_DEFAULT_NAMESPACE_END
