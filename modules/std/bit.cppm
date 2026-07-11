// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1

#include "dpl/config.h"

export module dpl:std.bit;
export import :std.stddef;
import :std.details.bit;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls)
using __DPL bit_cast;
using __DPL bit_ceil;
using __DPL bit_type;
using __DPL bit_type_t;
using __DPL bit_width;
using __DPL byteswap;
using __DPL char_bit_v;
using __DPL countl_one;
using __DPL countl_zero;
using __DPL countr_one;
using __DPL countr_zero;
using __DPL has_single_bit;
using __DPL popcount;
using __DPL rotl;
using __DPL rotr;
// NOLINTEND(misc-unused-using-decls)

__DPL_DEFAULT_NAMESPACE_END
