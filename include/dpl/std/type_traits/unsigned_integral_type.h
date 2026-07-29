// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/stddef/types.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t N>
struct unsigned_integral_type {};

template <size_t N>
using unsigned_integral_type_t = typename unsigned_integral_type<N>::type;

template <>
struct unsigned_integral_type<8> {
    using type DPL_NODEBUG = __DPL uint8;
};

template <>
struct unsigned_integral_type<16> {
    using type DPL_NODEBUG = __DPL uint16;
};

template <>
struct unsigned_integral_type<32> {
    using type DPL_NODEBUG = __DPL uint32;
};

template <>
struct unsigned_integral_type<64> {
    using type DPL_NODEBUG = __DPL uint64;
};

#if DPL_SUPPORTS_INT128
template <>
struct unsigned_integral_type<128> {
    using type DPL_NODEBUG = __DPL uint128;
};
#endif

__DPL_DEFAULT_NAMESPACE_END
