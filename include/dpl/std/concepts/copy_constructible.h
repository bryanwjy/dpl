// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/constructible_from.h"
#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/move_constructible.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept copy_constructible = move_constructible<T> &&
    constructible_from<T, T&> && convertible_to<T&, T> &&
    constructible_from<T, T const&> && convertible_to<T const&, T> &&
    constructible_from<T, T const> && convertible_to<T const, T>;

DPL_DEFAULT_NAMESPACE_END
