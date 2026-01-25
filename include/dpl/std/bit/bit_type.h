// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/stddef.h"
#  include "dpl/std/type_traits/make_signed.h"
#endif

#include "dpl/std/bit/char_bit.h"
#include "dpl/std/bit/has_single_bit.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <size_t W>
struct bit_type {};

DPL_EXPORT template <size_t W>
requires (W <= char_bit_v && __DPL has_single_bit(W))
struct bit_type<W> {
    using type = uint8;
};

DPL_EXPORT template <>
struct bit_type<16> {
    using type = uint16;
};

DPL_EXPORT template <>
struct bit_type<32> {
    using type = uint32;
};

DPL_EXPORT template <>
struct bit_type<64> {
    using type = uint64;
};

#if DPL_SUPPORTS_INT128
DPL_EXPORT template <>
struct bit_type<128> {
    using type = uint128;
};
#endif

DPL_EXPORT template <size_t W>
using bit_type_t = typename bit_type<W>::type;

DPL_DEFAULT_NAMESPACE_END
