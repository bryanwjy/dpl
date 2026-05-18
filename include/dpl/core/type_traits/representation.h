// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
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

DPL_EXPORT template <integral T>
struct signed_representation<T> {
    using type DPL_NODEBUG = make_signed_t<T>;
};
DPL_EXPORT template <enumeration T>
struct signed_representation<T> {
    using type DPL_NODEBUG = make_signed_t<underlying_type_t<T>>;
};
DPL_EXPORT template <typename T>
requires (!integral<T> && !enumeration<T>) && is_object_v<T> &&
    is_trivially_copyable_v<T> &&
    requires { typename bit_type_t<sizeof(T) * char_bit_v>; }
struct signed_representation<T> {
    using type DPL_NODEBUG = make_signed_t<bit_type_t<sizeof(T) * char_bit_v>>;
};

DPL_EXPORT template <typename T>
struct unsigned_representation {};
DPL_EXPORT template <typename T>
using unsigned_representation_t = typename unsigned_representation<T>::type;

DPL_EXPORT template <integral T>
struct unsigned_representation<T> {
    using type DPL_NODEBUG = make_unsigned_t<T>;
};
DPL_EXPORT template <enumeration T>
struct unsigned_representation<T> {
    using type DPL_NODEBUG = make_unsigned_t<underlying_type_t<T>>;
};
DPL_EXPORT template <typename T>
requires (!integral<T> && !enumeration<T>) && is_object_v<T> &&
    is_trivially_copyable_v<T> &&
    requires { typename bit_type_t<sizeof(T) * char_bit_v>; }
struct unsigned_representation<T> {
    using type DPL_NODEBUG =
        make_unsigned_t<bit_type_t<sizeof(T) * char_bit_v>>;
};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
