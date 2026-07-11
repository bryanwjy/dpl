// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1

#include "dpl/config.h"

export module dpl:std.details.concepts;
export import :std.stddef;
import :std.type_traits;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/std/concepts/array_initializable.h"
#include "dpl/std/concepts/assignable_from.h"
#include "dpl/std/concepts/boolean_testable.h"
#include "dpl/std/concepts/common_reference_with.h"
#include "dpl/std/concepts/common_with.h"
#include "dpl/std/concepts/constructible_from.h"
#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/copy_constructible.h"
#include "dpl/std/concepts/copyable.h"
#include "dpl/std/concepts/default_initializable.h"
#include "dpl/std/concepts/derived_from.h"
#include "dpl/std/concepts/destructible.h"
#include "dpl/std/concepts/different_from.h"
#include "dpl/std/concepts/enumeration.h"
#include "dpl/std/concepts/equality_comparable.h"
#include "dpl/std/concepts/floating_point.h"
#include "dpl/std/concepts/integer_sequence_like.h"
#include "dpl/std/concepts/integral.h"
#include "dpl/std/concepts/integral_constant_like.h"
#include "dpl/std/concepts/invocable.h"
#include "dpl/std/concepts/movable.h"
#include "dpl/std/concepts/move_constructible.h"
#include "dpl/std/concepts/regular.h"
#include "dpl/std/concepts/same_as.h"
#include "dpl/std/concepts/semiregular.h"
#include "dpl/std/concepts/swappable.h"
#include "dpl/std/concepts/totally_ordered.h"
#include "dpl/std/concepts/tuple_like.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
