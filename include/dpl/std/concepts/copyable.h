// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/assignable_from.h"
#include "dpl/std/concepts/copy_constructible.h"
#include "dpl/std/concepts/movable.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <class T>
concept copyable =
    copy_constructible<T> && movable<T> && assignable_from<T&, T&> &&
    assignable_from<T&, T const&> && assignable_from<T&, T const>;

__DPL_DEFAULT_NAMESPACE_END
