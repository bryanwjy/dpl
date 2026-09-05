// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.dispatch;
export import :core.fwd;
import :core.type_traits;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.concepts;
import :core.type_traits;
import :core.immediate;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/core/dispatch/concepts/operation.h"
#include "dpl/core/dispatch/evaluate.h"
#include "dpl/core/dispatch/interface.h"
#include "dpl/core/dispatch/maskable/accumulation.h"
#include "dpl/core/dispatch/maskable/predicate.h"
#include "dpl/core/dispatch/maskable/transform.h"
#include "dpl/core/dispatch/operation/algorithm.h"
#include "dpl/core/dispatch/operation/basic.h"
#include "dpl/core/dispatch/operation/math.h"
#include "dpl/core/dispatch/operation/primitive.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()