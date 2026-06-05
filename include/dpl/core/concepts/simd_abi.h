// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/derived_from.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/semiregular.h"
#  include "dpl/std/type_traits/is_empty.h"
#  include "dpl/std/type_traits/remove_cv.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename T>
struct simd_abi_base {
protected:
    constexpr ~simd_abi_base() = default;
};

DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_abi = derived_from<T, simd_abi_base<T>>;

namespace internal {
template <template <typename> typename>
struct unary_template;
template <typename T>
concept simd_abi =
    enable_simd_abi<T> && is_empty_v<T> && semiregular<T> && requires {
        typename integral_constant<T, T{}>;
        typename internal::unary_template<T::template native_vector>;
        typename internal::unary_template<T::template native_mask>;
    };
} // namespace internal

DPL_EXPORT template <typename T>
concept fixed_width_abi = internal::simd_abi<remove_cv_t<T>> &&
    requires { typename integral_constant<size_t, remove_cv_t<T>::size>; };

DPL_EXPORT template <typename T>
concept scalable_abi =
    internal::simd_abi<remove_cv_t<T>> && !fixed_width_abi<T> && requires {
        // Checking everything seems to be quite expensive
        { remove_cv_t<T>::template size<char>() } -> unsigned_integral;
    };

DPL_EXPORT template <typename T>
concept simd_abi = fixed_width_abi<T> || scalable_abi<T>;

namespace internal {
template <simd_abi A>
consteval remove_cv_t<A> make_abi() noexcept {
    return remove_cv_t<A>{};
}

template <typename T>
requires requires {
    typename remove_cv_t<T>::abi_type;
    { make_abi<typename remove_cv_t<T>::abi_type>() } -> simd_abi;
}
consteval typename remove_cv_t<T>::abi_type make_abi() noexcept {
    return typename remove_cv_t<T>::abi_type{};
}

template <typename A>
requires requires { internal::make_abi<A>(); }
inline constexpr auto abi = make_abi<A>();
} // namespace internal
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
