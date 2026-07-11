// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/constructible_from.h"
#include "dpl/std/concepts/convertible_to.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept move_constructible = constructible_from<T, T> && convertible_to<T, T>;

__DPL_DEFAULT_NAMESPACE_END
