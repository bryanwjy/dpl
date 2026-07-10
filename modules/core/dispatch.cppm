// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.dispatch;
export import :core.fwd;
import :core.details.type_traits;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.concepts;
import :core.type_traits;
import :core.immediate;

// IWYU pragma: begin_exports
#include "dpl/core/dispatch/broadcastable/base.h"
#include "dpl/core/dispatch/broadcastable/binary.h"
#include "dpl/core/dispatch/broadcastable/selection.h"
#include "dpl/core/dispatch/broadcastable/ternary.h"
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
