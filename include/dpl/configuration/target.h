/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_compiler.h"

#if DPL_COMPILER_CLANG

#  if defined(__CYGWIN__) || defined(_WIN32) || defined(_WIN64)
#    define DPL_TARGET_MICROSOFT 1
#  elif defined(__APPLE__)
#    define DPL_TARGET_APPLE 1
#  elif defined(RENAME_TO_SONY_IDENTIFIER) /* NDA */
#    define DPL_TARGET_SONY 1
#  elif defined(RENAME_TO_NINTENDO_IDENTIFIER) /* NDA */
#    define DPL_TARGET_NINTENDO 1
#  elif defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__) || \
      defined(__OpenBSD__)
#    define DPL_TARGET_BSD 1
#  elif defined(__linux__)
#    define DPL_TARGET_LINUX 1
#  elif defined(__unix__)
#    define DPL_TARGET_UNIX 1
#  else
#    error "Unrecognized target for Clang Compiler"
#  endif

#elif DPL_COMPILER_GCC

#  if defined(__CYGWIN__) || defined(_WIN32) || defined(_WIN64)
#    define DPL_TARGET_MICROSOFT 1
#  elif defined(__APPLE__)
#    define DPL_TARGET_APPLE 1
#  elif defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__) || \
      defined(__OpenBSD__)
#    define DPL_TARGET_BSD 1
#  elif defined(__linux__)
#    define DPL_TARGET_LINUX 1
#  elif defined(__unix__)
#    define DPL_TARGET_UNIX 1
#  else
#    error "Unrecognized target for GNU Compiler"
#  endif

#elif DPL_COMPILER_MSVC

#  define DPL_TARGET_MICROSOFT 1

#elif DPL_COMPILER_ICC || DPL_COMPILER_ICX

#  if defined(__CYGWIN__) || defined(_WIN32) || defined(_WIN64)
#    define DPL_TARGET_MICROSOFT 1
#  elif defined(__APPLE__)
#    define DPL_TARGET_APPLE 1
#  elif defined(__FreeBSD__) || defined(__DragonFly__) || defined(__NetBSD__) || \
      defined(__OpenBSD__)
#    define DPL_TARGET_BSD 1
#  elif defined(__linux__)
#    define DPL_TARGET_LINUX 1
#  elif defined(__unix__)
#    define DPL_TARGET_UNIX 1
#  else
#    error "Unrecognized target for Intel Compiler"
#  endif

#endif /* if DPL_COMPILER_CLANG */

#if DPL_TARGET_APPLE && !DPL_TARGET_UNIX
#  define DPL_TARGET_UNIX 1
#endif /* if DPL_TARGET_APPLE && !defined(DPL_TARGET_UNIX) */

#if DPL_TARGET_LINUX && !DPL_TARGET_UNIX
#  define DPL_TARGET_UNIX 1
#endif /* if DPL_TARGET_LINUX && !defined(DPL_TARGET_UNIX) */

#if DPL_TARGET_SONY
#  if !DPL_TARGET_BSD
#    define DPL_TARGET_BSD 1
#  endif
#  if !DPL_TARGET_UNIX
#    define DPL_TARGET_UNIX 1
#  endif
#endif /* if DPL_TARGET_SONY && !defined(DPL_TARGET_BSD) */

#if DPL_TARGET_BSD && !DPL_TARGET_UNIX
#  define DPL_TARGET_UNIX 1
#endif /* if DPL_TARGET_BSD && !defined(DPL_TARGET_UNIX) */

#if DPL_TARGET_SONY
// AFAIK the Pro models are not dedicated targets, I could be wrong though
#  if RENAME_TO_PS4_IDENTIFIER /* NDA */
#    define DPL_TARGET_SONY_PS4 1
#  elif RENAME_TO_PS5_IDENTIFIER /* NDA */
#    define DPL_TARGET_SONY_PS5 1
#  else
#    error "Unrecognized Sony target"
#  endif
#endif /* if DPL_TARGET_SONY */

#if DPL_TARGET_MICROSOFT

#  if defined(__CYGWIN__)
#    define DPL_TARGET_MICROSOFT_WINDOWS 1
#    define DPL_TARGET_MICROSOFT_CYGWIN 1
#  elif defined(_WIN32) || defined(_WIN64)
#    define DPL_TARGET_MICROSOFT_WINDOWS 1
#  elif RENAME_TO_GDK_XBO_IDENTIFIER /* NDA */
#    define DPL_TARGET_MICROSOFT_GDK_XBO 1
#  elif RENAME_TO_GDK_XSX_IDENTIFIER /* NDA */
#    define DPL_TARGET_MICROSOFT_GDK_XSX 1
#  else
#    error "Unrecognized Microsoft target"
#  endif

#  if RENAME_TO_DESKTOP_GDK_IDENTIFIER /* NDA */
#    define DPL_TARGET_MICROSOFT_GDK_WINDOWS 1
#  endif

#endif /* if DPL_TARGET_MICROSOFT */

#if DPL_COMPILER_CLANG | DPL_COMPILER_GCC | DPL_COMPILER_ICX | \
    (DPL_COMPILER_ICC & !DPL_TARGET_MICROSOFT)
#  define DPL_SUPPORTS_GNU_ASM 1
#endif
