// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.algorithm;
export import :core.fwd;
export import :core.operations;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.basic;
import :core.concepts;
import :core.type_traits;
import :core.utility;

// IWYU pragma: begin_exports
#include "dpl/core/algorithm/clamp.h"
#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/cswap.h"
#include "dpl/core/algorithm/hmax.h"
#include "dpl/core/algorithm/hmin.h"
#include "dpl/core/algorithm/hsum.h"
#include "dpl/core/algorithm/inner_product.h"
#include "dpl/core/algorithm/reduce.h"
#include "dpl/core/algorithm/rotate.h"
#include "dpl/core/algorithm/scan.h"
#include "dpl/core/algorithm/scan_max.h"
#include "dpl/core/algorithm/scan_min.h"
#include "dpl/core/algorithm/scan_sum.h"
#include "dpl/core/algorithm/shift.h"
#include "dpl/core/algorithm/slide.h"
#include "dpl/core/algorithm/splice.h"
// IWYU pragma: end_exports
