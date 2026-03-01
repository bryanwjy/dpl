// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/std/stddef/namespace.h" // IWYU pragma: export

#if !DPL_MODULES
#  include <stdint.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT using ptrdiff_t = decltype((char*)0 - (char*)0);
DPL_EXPORT using size_t = decltype(sizeof(0));

#if DPL_SUPPORTS_FLOAT16
DPL_EXPORT using float16 = decltype(0.0f16);
#endif
#if DPL_SUPPORTS_FLOAT32
DPL_EXPORT using float32 = decltype(0.0f32);
#endif
#if DPL_SUPPORTS_FLOAT64
DPL_EXPORT using float64 = decltype(0.0f64);
#endif
#if DPL_SUPPORTS_FLOAT128
DPL_EXPORT using float128 = decltype(0.0f128);
#endif
#if DPL_SUPPORTS_BFLOAT16
DPL_EXPORT using bfloat16 = decltype(0.0bf16);
#endif

#if DPL_SUPPORTS_INT128
DPL_EXPORT using int128 = __int128_t;
DPL_EXPORT using uint128 = __uint128_t;
#endif

DPL_EXPORT using int64 = int64_t;
DPL_EXPORT using int32 = int32_t;
DPL_EXPORT using int16 = int16_t;
DPL_EXPORT using int8 = int8_t;
DPL_EXPORT using uint64 = uint64_t;
DPL_EXPORT using uint32 = uint32_t;
DPL_EXPORT using uint16 = uint16_t;
DPL_EXPORT using uint8 = uint8_t;

DPL_DEFAULT_NAMESPACE_END
