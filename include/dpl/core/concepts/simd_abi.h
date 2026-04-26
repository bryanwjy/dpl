// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/semiregular.h"
#  include "dpl/std/type_traits/is_empty.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_abi = false;
DPL_EXPORT inline constexpr size_t scalable_size = -1zu;

namespace internal {
template <template <typename> typename>
struct unary_template;
template <auto>
struct constant_value;
} // namespace internal

DPL_EXPORT template <typename T>
concept simd_abi =
    enable_simd_abi<T> && is_empty_v<T> && semiregular<T> && requires {
        typename internal::constant_value<T{}>;
        typename internal::unary_template<T::template native_type>;
        typename internal::unary_template<T::template native_mask>;
    };

DPL_EXPORT template <typename T>
concept fixed_width_abi = simd_abi<T> && requires {
    typename internal::constant_value<T::size>;
    typename internal::constant_value<T::alignment>;
} && (T::size != scalable_size);

DPL_EXPORT template <typename T>
concept scalable_abi = !fixed_width_abi<T> && simd_abi<T> &&
    (T::size == scalable_size) && requires {
        typename internal::constant_value<T::max_size>;
        // TODO: element_count
    };

namespace internal {
template <simd_abi A>
consteval A make_abi() noexcept {
    return A{};
}

template <typename T>
requires requires {
    typename T::abi_type;
    { make_abi<typename T::abi_type>() } -> simd_abi;
}
consteval typename T::abi_type make_abi() noexcept {
    return typename T::abi_type{};
}

template <typename A>
requires requires { internal::make_abi<A>(); }
inline constexpr auto abi = make_abi<A>();
} // namespace internal
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
