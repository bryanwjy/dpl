// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1

#include "dpl/config.h"

export module dpl:std.details.bit;
export import :std.stddef;
import :std.concepts;
import :std.type_traits;

// IWYU pragma: begin_exports
#include "dpl/std/bit/bit_cast.h"
#include "dpl/std/bit/bit_ceil.h"
#include "dpl/std/bit/bit_type.h"
#include "dpl/std/bit/bit_width.h"
#include "dpl/std/bit/byteswap.h"
#include "dpl/std/bit/char_bit.h"
#include "dpl/std/bit/countl.h"
#include "dpl/std/bit/countr.h"
#include "dpl/std/bit/has_single_bit.h"
#include "dpl/std/bit/popcount.h"
#include "dpl/std/bit/rotate.h"
// IWYU pragma: end_exports
