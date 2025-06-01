// Copyright 2025 Bryan Wong

#pragma once

// IWYU pragma: always_keep

// IWYU pragma: begin_exports
#include "dpl/configuration/abi.h"              // IWYU pragma: keep
#include "dpl/configuration/allocation.h"       // IWYU pragma: keep
#include "dpl/configuration/architecture.h"     // IWYU pragma: keep
#include "dpl/configuration/attributes.h"       // IWYU pragma: keep
#include "dpl/configuration/build_type.h"       // IWYU pragma: keep
#include "dpl/configuration/builtins.h"         // IWYU pragma: keep
#include "dpl/configuration/compiler.h"         // IWYU pragma: keep
#include "dpl/configuration/compiler_barrier.h" // IWYU pragma: keep
#include "dpl/configuration/exceptions.h"       // IWYU pragma: keep
#include "dpl/configuration/keywords.h"         // IWYU pragma: keep
#include "dpl/configuration/memcmp.h"           // IWYU pragma: keep
#include "dpl/configuration/memcpy.h"           // IWYU pragma: keep
#include "dpl/configuration/modules.h"          // IWYU pragma: keep
#include "dpl/configuration/namespace.h"        // IWYU pragma: keep
#include "dpl/configuration/pragma.h"           // IWYU pragma: keep
#include "dpl/configuration/simd.h"             // IWYU pragma: keep
#include "dpl/configuration/standard.h"         // IWYU pragma: keep
#include "dpl/configuration/target.h"           // IWYU pragma: keep
#include "dpl/configuration/types.h"            // IWYU pragma: keep
#include "dpl/preprocessor/attribute_list.h"    // IWYU pragma: keep
// IWYU pragma: end_exports

#if DPL_ARCH_x86_64
#  ifndef DPL_DEFAULT_SIMD_REGISTER
#    define DPL_DEFAULT_SIMD_REGISTER xmm
#    define __DPL_SIMD_ABI_INLINE_xmm inline
#    define __DPL_SIMD_ABI_INLINE_ymm
#    define __DPL_SIMD_ABI_INLINE_zmm
#    define __DPL_INLINE_IF_DEFAULT_SIMD_ABI(X) \
        DPL_CONCAT(__DPL_SIMD_ABI_INLINE_, X)

#  else // ifndef DPL_DEFAULT_SIMD_REGISTER
#    define __DPL_SIMD_REGISTER_xmm
#    define __DPL_SIMD_REGISTER_ymm
#    define __DPL_SIMD_REGISTER_zmm
#    define __DPL_HAS_SIMD_REGISTER(X) \
        DPL_IS_EMPTY(DPL_CONCAT(__DPL_REGISTER_, X))

#    define __DPL_SIMD_ABI_INLINE_0
#    define __DPL_SIMD_ABI_INLINE_1 inline
#    define __DPL_INLINE_IF_DEFAULT_SIMD_ABI(X)                        \
        DPL_CONCAT(__DPL_SIMD_ABI_INLINE_, __DPL_HAS_SIMD_REGISTER(X)) \
        DPL_CONCAT(__DPL_REGISTER_, X)

#    if !__DPL_HAS_SIMD_REGISTER(DPL_DEFAULT_SIMD_REGISTER)
#      error Invalid SIMD register specified as default
#    endif

#  endif // ifndef DPL_DEFAULT_SIMD_REGISTER

#  define DPL_SIMD_ABI_NAMESPACE_BEGIN(X) \
      __DPL_INLINE_IF_DEFAULT_SIMD_ABI(X) namespace X {
#  define DPL_SIMD_ABI_NAMESPACE_END(X) }

#endif // if DPL_ARCH_x86_64

#ifndef DPL_NS
#  define DPL_NS dpl
#endif

#define __DPL ::DPL_NS::

#define DPL_DEFAULT_NAMESPACE_BEGIN DPL_NAMESPACE_BEGIN(DPL_NS)

#define DPL_DEFAULT_NAMESPACE_END }

#define DPL_EXPORTED_DEFAULT_NAMESPACE_BEGIN \
    DPL_EXPORT DPL_DEFAULT_NAMESPACE_BEGIN

#define DPL_EXPORTED_DEFAULT_NAMESPACE_END DPL_DEFAULT_NAMESPACE_END

DPL_EXPORTED_DEFAULT_NAMESPACE_BEGIN

using ptrdiff_t = decltype((char*)0 - (char*)0);
using size_t = decltype(sizeof(0));

#if DPL_SUPPORTS_FLOAT16
using float16 = decltype(0.0f16);
#endif
#if DPL_SUPPORTS_FLOAT32
using float32 = decltype(0.0f32);
#endif
#if DPL_SUPPORTS_FLOAT64
using float64 = decltype(0.0f64);
#endif
#if DPL_SUPPORTS_FLOAT128
using float128 = decltype(0.0f128);
#endif
#if DPL_SUPPORTS_BFLOAT16
using bfloat16 = decltype(0.0bf16);
#endif

DPL_EXPORTED_DEFAULT_NAMESPACE_END
