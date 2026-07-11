// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/enable_simd_abi.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/semiregular.h"
#  include "dpl/std/type_traits/is_empty.h"
#  include "dpl/std/type_traits/remove_cv.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace atom {
template <template <typename> typename>
struct unary_template;
template <typename T>
concept simd_abi =
    enable_simd_abi<T> && is_empty_v<T> && semiregular<T> && requires {
        typename integral_constant<T, T{}>;
        typename unary_template<T::template native_vector>;
        typename unary_template<T::template native_mask>;
    };

template <typename T>
concept fixed_width_abi =
    requires { typename integral_constant<size_t, T::size>; };

template <typename T>
concept scalable_abi = !fixed_width_abi<T> && requires {
    // Check a few usual suspects
    { T::template size<char>() } -> core_convertible_to<size_t>;
    { T::template size<int>() } -> core_convertible_to<size_t>;
    { T::template size<float>() } -> core_convertible_to<size_t>;
};
} // namespace atom

template <typename T>
concept simd_abi = atom::simd_abi<remove_cv_t<T>> &&
    (atom::fixed_width_abi<remove_cv_t<T>> ||
        atom::scalable_abi<remove_cv_t<T>>);

template <typename T>
concept fixed_width_abi = simd_abi<T> && atom::fixed_width_abi<remove_cv_t<T>>;

// Unfortunately, compilers currently (2026) disallows encapsulating sizeless
// types References to scalable types are only here for completeness and not
// supported
template <typename T>
concept scalable_abi = simd_abi<T> && atom::scalable_abi<remove_cv_t<T>>;

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
