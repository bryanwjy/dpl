/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/configuration/compiler.h" // IWYU pragma: keep

#ifndef _LIBCPP_HAS_NO_EXCEPTIONS
#  if defined(__cpp_exceptions)
#    define DPL_WITH_EXCEPTIONS 1
#  elif DPL_COMPILER_MSVC && defined(_CPPUNWIND)
#    if _CPPUNWIND
#      define DPL_WITH_EXCEPTIONS 1
#    endif
#  elif defined(__EXCEPTIONS)
#    define DPL_WITH_EXCEPTIONS 1
#  endif
#endif
