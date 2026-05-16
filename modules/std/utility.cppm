// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:std.utility;
export import :std.stddef;
import :std.type_traits;
import :std.concepts;

// IWYU pragma: begin_exports
#include "dpl/std/utility/apply.h"
#include "dpl/std/utility/as_const.h"
#include "dpl/std/utility/exchange.h"
#include "dpl/std/utility/forward.h"
#include "dpl/std/utility/forward_like.h"
#include "dpl/std/utility/move.h"
#include "dpl/std/utility/sequence.h"
#include "dpl/std/utility/structured_bindings.h"
#include "dpl/std/utility/template_barrier.h"
#include "dpl/std/utility/to_signed.h"
#include "dpl/std/utility/to_underlying.h"
#include "dpl/std/utility/to_unsigned.h"
#include "dpl/std/utility/tuple_like.h"
#include "dpl/std/utility/type_pack.h"
#include "dpl/std/utility/unreachable.h"
#include "dpl/std/utility/value_pack.h"
// IWYU pragma: end_exports
