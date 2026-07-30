// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.utility;
export import :core.fwd;
import :core.type_traits;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.basic;
import :core.concepts;
import :core.immediate;
import :core.operations;
import :core.type_traits;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/core/utility/swap.h"
#include "dpl/core/utility/to_canonical.h"
#include "dpl/core/utility/to_signed.h"
#include "dpl/core/utility/to_tuple_like.h"
#include "dpl/core/utility/to_unsigned.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
