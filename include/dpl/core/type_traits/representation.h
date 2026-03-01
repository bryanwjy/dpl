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
struct signed_representation {};

DPL_EXPORT template <typename T>
using signed_representation_t = typename signed_representation<T>::type;

DPL_EXPORT template <simd_element T>
requires integral<T>
struct signed_representation<T> {
    using type DPL_NODEBUG = make_signed_t<T>;
};
DPL_EXPORT template <simd_element T>
requires enumeration<T>
struct signed_representation<T> {
    using type DPL_NODEBUG = make_signed_t<underlying_type_t<T>>;
};
DPL_EXPORT template <simd_element T>
struct signed_representation<T> {
    using type DPL_NODEBUG = make_signed_t<bit_type_t<sizeof(T) * char_bit_v>>;
};

DPL_EXPORT template <typename T>
struct unsigned_representation {};
DPL_EXPORT template <typename T>
using unsigned_representation_t = typename unsigned_representation<T>::type;

DPL_EXPORT template <simd_element T>
requires integral<T>
struct unsigned_representation<T> {
    using type DPL_NODEBUG = make_unsigned_t<T>;
};
DPL_EXPORT template <simd_element T>
requires enumeration<T>
struct unsigned_representation<T> {
    using type DPL_NODEBUG = make_unsigned_t<underlying_type_t<T>>;
};
DPL_EXPORT template <simd_element T>
struct unsigned_representation<T> {
    using type DPL_NODEBUG =
        make_unsigned_t<bit_type_t<sizeof(T) * char_bit_v>>;
};

namespace internal {
template <typename T>
using signed_rep DPL_NODEBUG = signed_representation<T>;
template <typename T>
using signed_rep_t DPL_NODEBUG = signed_representation_t<T>;
template <typename T>
using unsigned_rep DPL_NODEBUG = unsigned_representation<T>;
template <typename T>
using unsigned_rep_t DPL_NODEBUG = unsigned_representation_t<T>;
} // namespace internal

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
