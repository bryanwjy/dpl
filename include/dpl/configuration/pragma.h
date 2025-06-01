/* Copyright 2023-2025 Bryan Wong */

#include "configuration/utl_compiler.h"
#include "configuration/utl_target.h"

#if DPL_COMPILER_MSVC && !DPL_COMPILER_MSVC_AT_LEAST(1925)
#  define DPL_PRAGMA(ARG) __pragma(ARG)
#else
#  define DPL_PRAGMA(ARG) _Pragma(#ARG)
#endif

#if DPL_TARGET_APPLE
/* Apple supposedly silently ignores the region token */
#  define DPL_SUPPORTS_PRAGMA_REGION
#  define DPL_BEGIN_REGION(LABEL) DPL_PRAGMA(region LABEL)
#  define DPL_END_REGION(LABEL) DPL_PRAGMA(endregion LABEL)
#elif DPL_COMPILER_CLANG_AT_LEAST(6, 0, 0)
#  define DPL_SUPPORTS_PRAGMA_REGION
#  define DPL_BEGIN_REGION(LABEL) DPL_PRAGMA(region LABEL)
#  define DPL_END_REGION(LABEL) DPL_PRAGMA(endregion LABEL)
#elif DPL_COMPILER_GCC_AT_LEAST(13, 0, 0)
#  define DPL_SUPPORTS_PRAGMA_REGION
#  define DPL_BEGIN_REGION(LABEL) DPL_PRAGMA(region LABEL)
#  define DPL_END_REGION(LABEL) DPL_PRAGMA(endregion LABEL)
#elif DPL_COMPILER_MSVC
#  define DPL_SUPPORTS_PRAGMA_REGION
#  define DPL_BEGIN_REGION(LABEL) DPL_PRAGMA(region LABEL)
#  define DPL_END_REGION(LABEL) DPL_PRAGMA(endregion LABEL)
#else
#  define DPL_BEGIN_REGION(_)
#  define DPL_END_REGION(_)
#endif

#if DPL_COMPILER_CLANG || DPL_COMPILER_ICX

#  define DPL_PRAGMA_WARN(x) DPL_PRAGMA(message #x)

#elif DPL_COMPILER_GCC

#  if DPL_COMPILER_GCC_AT_LEAST(9, 1, 0)
#    define DPL_PRAGMA_WARN(x) DPL_PRAGMA(GCC warning #x)
#  else /* DPL_COMPILER_GCC_AT_LEAST(9,1,0) */
#    define DPL_PRAGMA_WARN(x) DPL_PRAGMA(GCC message #x)
#  endif

#elif DPL_COMPILER_MSVC

#  define DPL_WARN_PRIVATE_STR2(x) #x
#  define DPL_WARN_PRIVATE_STR1(x) DPL_WARN_PRIVATE_STR2(x)
#  define DPL_WARN_PRIVATE_LOCATION __FILE__ ":" DPL_WARN_PRIVATE_STR1(__LINE__) ":"
#  define DPL_PRAGMA_WARN(x) DPL_PRAGMA(message(DPL_WARN_PRIVATE_LOCATION " warning: " #x))

#else

#  define DPL_PRAGMA_WARN(_)

#endif
/* TODO: Find out ICC's warning pragma */

#if DPL_COMPILER_MSVC | DPL_COMPILER_ICC
#  define DPL_DISABLE_WARNING_PUSH() DPL_PRAGMA(warning(push))
#  define DPL_DISABLE_WARNING_POP() DPL_PRAGMA(warning(pop))
#  define DPL_DISABLE_WARNING(warningNumber) DPL_PRAGMA(warning(disable : warningNumber))
#elif DPL_COMPILER_CLANG | DPL_COMPILER_ICX
#  define DPL_DISABLE_WARNING_PUSH() DPL_PRAGMA(clang diagnostic push)
#  define DPL_DISABLE_WARNING_POP() DPL_PRAGMA(clang diagnostic pop)
#  define DPL_DISABLE_WARNING(warningName) DPL_PRAGMA(clang diagnostic ignored warningName)
#elif DPL_COMPILER_GCC
#  define DPL_DISABLE_WARNING_PUSH() DPL_PRAGMA(GCC diagnostic push)
#  define DPL_DISABLE_WARNING_POP() DPL_PRAGMA(GCC diagnostic pop)
#  define DPL_DISABLE_WARNING(warningName) DPL_PRAGMA(GCC diagnostic ignored warningName)
#else
#  define DPL_DISABLE_WARNING_PUSH()
#  define DPL_DISABLE_WARNING_POP()
#  define DPL_DISABLE_WARNING(warningName)
#endif
