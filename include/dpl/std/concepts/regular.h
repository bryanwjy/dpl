// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/equality_comparable.h"
#include "dpl/std/concepts/semiregular.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_invocable.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN template <typename T>
concept regular = semiregular<T> && equality_comparable<T>;
__DPL_DEFAULT_NAMESPACE_END
