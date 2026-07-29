// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/stddef/types.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t N>
struct signed_integral_type {};

template <size_t N>
using signed_integral_type_t = typename signed_integral_type<N>::type;

template <>
struct signed_integral_type<8> {
    using type DPL_NODEBUG = __DPL int8;
};

template <>
struct signed_integral_type<16> {
    using type DPL_NODEBUG = __DPL int16;
};

template <>
struct signed_integral_type<32> {
    using type DPL_NODEBUG = __DPL int32;
};

template <>
struct signed_integral_type<64> {
    using type DPL_NODEBUG = __DPL int64;
};

#if DPL_SUPPORTS_INT128
template <>
struct signed_integral_type<128> {
    using type DPL_NODEBUG = __DPL int128;
};
#endif

__DPL_DEFAULT_NAMESPACE_END
