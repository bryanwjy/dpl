// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/type_traits/element_count.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/type_traits/is_invocable.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename F, typename R, typename... Args>
concept value_invocable_r =
    (basic_simd_class<R> && ... && basic_simd_class<Args>) &&
    (... && (element_count<R> == element_count<Args>)) &&
    is_invocable_r_v<typename R::value_type, F, typename Args::value_type...>;

template <typename R, typename F, typename... Args, size_t... Is>
consteval bool invocable(index_sequence<Is...>) noexcept {
    return (... &&
        is_invocable_r_v<typename R::value_type, F, immediate<Is>,
            typename Args::value_type...>);
}

template <typename F, typename R, typename... Args>
concept ivalue_invocable_r =
    (basic_simd_class<R> && ... && basic_simd_class<Args>) &&
    (... && (element_count<R> == element_count<Args>)) &&
    internal::invocable<R, F, Args...>(iota_sequence<R>);

template <basic_simd_class Result, basic_simd_class... Ts,
    ivalue_invocable_r<Result, Ts...> Op>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, FLATTEN, NODISCARD)
constexpr Result itransform(Op func, Ts... args) noexcept {

    constexpr auto single = []<size_t I>(
                                immediate<I> idx, Ts... args, Op func) {
        return __DPL invoke_r<typename Result::value_type>(
            func, idx, args[idx]...);
    };

    return [single]<size_t... Is>(Ts... args, Op func, index_sequence<Is...>) {
        return Result{single(imm<Is>, args..., func)...};
    }(args..., func, iota_sequence<Result>);
}

template <basic_simd_class Result, basic_simd_class... Ts,
    value_invocable_r<Result, Ts...> Op>
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
