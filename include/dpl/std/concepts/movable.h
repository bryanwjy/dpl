// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/assignable_from.h"
#include "dpl/std/concepts/move_constructible.h"
#include "dpl/std/concepts/swappable.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_object.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept movable = is_object_v<T> && move_constructible<T> &&
    assignable_from<T&, T> && swappable<T>;

DPL_DEFAULT_NAMESPACE_END
