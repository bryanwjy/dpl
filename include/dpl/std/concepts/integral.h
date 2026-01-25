// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_integral.h"
#  include "dpl/std/type_traits/is_signed.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept integral = is_integral_v<T>;

DPL_EXPORT template <typename T>
concept signed_integral = integral<T> && is_signed_v<T>;

DPL_EXPORT template <typename T>
concept unsigned_integral = integral<T> && !signed_integral<T>;

DPL_DEFAULT_NAMESPACE_END
