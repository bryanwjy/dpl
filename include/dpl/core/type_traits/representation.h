// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/type_traits/signed_integral_type.h"
#  include "dpl/std/type_traits/unsigned_integral_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
struct signed_representation {};

template <typename T>
using signed_representation_t = typename signed_representation<T>::type;

template <typename T>
requires requires { typename signed_integral_type_t<__DPL type_bit_v<T>>; }
struct signed_representation<T> :
    signed_integral_type<__DPL type_bit_v<T>> {};

template <typename T>
struct unsigned_representation {};
template <typename T>
using unsigned_representation_t = typename unsigned_representation<T>::type;

template <typename T>
requires requires { typename unsigned_integral_type_t<__DPL type_bit_v<T>>; }
struct unsigned_representation<T> :
    unsigned_integral_type<__DPL type_bit_v<T>> {};
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
