// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1

#include "dpl/config.h"

export module dpl:std.bit;
export import :std.stddef;
import :std.details.bit;

export DPL_DEFAULT_NAMESPACE_BEGIN

using __VDPL bit_cast;
using __VDPL bit_ceil;
using __VDPL bit_type;
using __VDPL bit_type_t;
using __VDPL bit_width;
using __VDPL byteswap;
using __VDPL char_bit_v;
using __VDPL countl_one;
using __VDPL countl_zero;
using __VDPL countr_one;
using __VDPL countr_zero;
using __VDPL has_single_bit;
using __VDPL popcount;
using __VDPL rotl;
using __VDPL rotr;

DPL_DEFAULT_NAMESPACE_END
