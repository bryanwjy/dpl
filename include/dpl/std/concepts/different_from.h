// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/same_as.h"

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT template <typename T, typename U>
concept different_from = !same_as<T, U>;
DPL_DEFAULT_NAMESPACE_END
