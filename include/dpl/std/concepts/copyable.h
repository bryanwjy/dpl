// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/assignable_from.h"
#include "dpl/std/concepts/copy_constructible.h"
#include "dpl/std/concepts/movable.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <class T>
concept copyable =
    copy_constructible<T> && movable<T> && assignable_from<T&, T&> &&
    assignable_from<T&, T const&> && assignable_from<T&, T const>;

DPL_DEFAULT_NAMESPACE_END
