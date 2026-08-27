// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.operations;
export import :core.fwd;
import :core.type_traits;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.basic;
import :core.concepts;
import :core.type_traits;
import :core.immediate;
import :core.dispatch;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/core/operations/abi_promotion.h"
#include "dpl/core/operations/arithmetic.h"
#include "dpl/core/operations/bit.h"
#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/broadcast_lane.h"
#include "dpl/core/operations/cast.h"
#include "dpl/core/operations/compare.h"
#include "dpl/core/operations/concat.h"
#include "dpl/core/operations/internal/array_for.h"
#include "dpl/core/operations/internal/reduction.h"
#include "dpl/core/operations/internal/transform.h"
#include "dpl/core/operations/logical.h"
#include "dpl/core/operations/minmax.h"
#include "dpl/core/operations/permute.h"
#include "dpl/core/operations/reinterpret.h"
#include "dpl/core/operations/select.h"
#include "dpl/core/operations/split.h"
#include "dpl/core/operations/split_result.h"
#include "dpl/core/operations/to_simd_mask.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
