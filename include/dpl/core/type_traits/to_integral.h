// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/type_traits/make_signed.h"
#  include "dpl/std/type_traits/make_unsigned.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct to_signed_integral {};
DPL_EXPORT template <typename T>
using to_signed_integral_t = typename to_signed_integral<T>::type;

DPL_EXPORT template <simd_element T>
requires integral<T>
struct to_signed_integral<T> {
    using type DPL_NODEBUG = T;
};
DPL_EXPORT template <simd_element T>
requires enumeration<T>
struct to_signed_integral<T> {
    using type DPL_NODEBUG = underlying_type_t<T>;
};
DPL_EXPORT template <simd_element T>
struct to_signed_integral<T> {
    using type DPL_NODEBUG = make_signed_t<bit_type_t<sizeof(T) * char_bit_v>>;
};

DPL_EXPORT template <typename T>
struct to_unsigned_integral {};
DPL_EXPORT template <typename T>
using to_unsigned_integral_t = typename to_unsigned_integral<T>::type;

DPL_EXPORT template <simd_element T>
requires integral<T>
struct to_unsigned_integral<T> {
    using type DPL_NODEBUG = T;
};
DPL_EXPORT template <simd_element T>
requires enumeration<T>
struct to_unsigned_integral<T> {
    using type DPL_NODEBUG = underlying_type_t<T>;
};
DPL_EXPORT template <simd_element T>
struct to_unsigned_integral<T> {
    using type DPL_NODEBUG =
        make_unsigned_t<bit_type_t<sizeof(T) * char_bit_v>>;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
