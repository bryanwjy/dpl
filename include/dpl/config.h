// Copyright 2025 Bryan Wong

#pragma once

// IWYU pragma: always_keep

// IWYU pragma: begin_exports
#include "dpl/configuration/abi.h"
#include "dpl/configuration/allocation.h"
#include "dpl/configuration/architecture.h"
#include "dpl/configuration/attributes.h"
#include "dpl/configuration/builtins.h"
#include "dpl/configuration/compiler.h"
#include "dpl/configuration/compiler_barrier.h"
#include "dpl/configuration/exceptions.h"
#include "dpl/configuration/keywords.h"
#include "dpl/configuration/memcmp.h"
#include "dpl/configuration/memcpy.h"
#include "dpl/configuration/modules.h"
#include "dpl/configuration/namespace.h"
#include "dpl/configuration/pragma.h"
#include "dpl/configuration/simd.h"
#include "dpl/configuration/standard.h"
#include "dpl/configuration/stl.h"
#include "dpl/configuration/target.h"
#include "dpl/configuration/types.h"
#include "dpl/preprocessor/attribute_list.h"
// IWYU pragma: end_exports

#if !DPL_CXX23
#  error "C++23 requirement not met"
#endif

#ifndef DPL_NS
#  define DPL_NS dpl
#endif

#define __DPL ::DPL_NS::

#define DPL_DEFAULT_NAMESPACE_BEGIN DPL_NAMESPACE_BEGIN(DPL_NS)

#define DPL_DEFAULT_NAMESPACE_END DPL_NAMESPACE_END(DPL_NS)

#if DPL_WITH_EXCEPTIONS

#  define DPL_THROW(...) throw(__VA_ARGS__)
#  define DPL_RETHROW(...) throw
#  define DPL_TRY try
#  define DPL_CATCH(...) catch (__VA_ARGS__)
#else
#  include <cassert>

#  define DPL_THROW(...)                        \
      []() {                                    \
        assert([]<typename T>() {               \
          return sizeof(T) != sizeof(T);        \
        }.operator()<decltype(__VA_ARGS__)>()); \
        DPL_BUILTIN_unreachable();              \
      }()
#  define DPL_RETHROW(...) \
      assert(false);       \
      DPL_BUILTIN_unreachable()

#  define DPL_TRY if constexpr (true)

#  define DPL_CATCH(...)                               \
      else if constexpr ([](auto&&) { return false; }( \
                             [](__VA_ARGS__) -> void {}))

#endif
