// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1

#include "dpl/config.h"

export module dpl:std.concepts;
import :std.stddef;
import :std.type_traits;

// IWYU pragma: begin_exports
#include "dpl/std/concepts/assignable_from.h"
#include "dpl/std/concepts/common_reference_with.h"
#include "dpl/std/concepts/constructible_from.h"
#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/copy_constructible.h"
#include "dpl/std/concepts/copyable.h"
#include "dpl/std/concepts/default_initializable.h"
#include "dpl/std/concepts/destructible.h"
#include "dpl/std/concepts/floating_point.h"
#include "dpl/std/concepts/integral.h"
#include "dpl/std/concepts/movable.h"
#include "dpl/std/concepts/move_constructible.h"
#include "dpl/std/concepts/same_as.h"
#include "dpl/std/concepts/semiregular.h"
#include "dpl/std/concepts/swappable.h"
// IWYU pragma: end_exports
