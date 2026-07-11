// Copyright 2025 Bryan Wong

#ifndef DPL_CONFIG_H
#define DPL_CONFIG_H

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
#  define DPL_IS_STD 0
#elif !DPL_DISABLE_HOST_STL_SUPPORT
#  define DPL_IS_STD 0
static_assert(
    [](int DPL_NS) {
#  define std 1
        return DPL_NS != 1;
#  undef std
    }(0),
    "Namespace cannot be 'std' while host stl support is enabled");
#else
DPL_DISABLE_WARNING_PUSH()
#  if DPL_COMPILER_GCC | DPL_COMPILER_CLANG
DPL_DISABLE_WARNING("-Wundef")
#  endif
#  define __DPL_NAMESPACE_CHECK_std 1
#  if DPL_CONCAT(__DPL_NAMESPACE_CHECK_, DPL_NS)
#    define DPL_IS_STD 1
#  endif
#  undef __DPL_NAMESPACE_CHECK_std
DPL_DISABLE_WARNING_POP()
#endif

#define __DPL ::DPL_NS::

#define DPL_DEFAULT_NAMESPACE_BEGIN DPL_NAMESPACE_BEGIN(DPL_NS)

#define DPL_DEFAULT_NAMESPACE_END DPL_NAMESPACE_END(DPL_NS)
#define __DPL_ABI_NAMESPACE abi_v0

#if DPL_MODULES
#  define __DPL_DEFAULT_NAMESPACE_BEGIN \
      export DPL_NAMESPACE_BEGIN(DPL_NS) namespace __DPL_ABI_NAMESPACE {

#  define __DPL_DEFAULT_NAMESPACE_END \
      DPL_NAMESPACE_END(DPL_NS)       \
      }
#else
#  define __DPL_DEFAULT_NAMESPACE_BEGIN \
      export DPL_NAMESPACE_BEGIN(DPL_NS) inline namespace __DPL_ABI_NAMESPACE {

#  define __DPL_DEFAULT_NAMESPACE_END \
      DPL_NAMESPACE_END(DPL_NS)       \
      }
#endif

#define __VDPL ::DPL_NS::__DPL_ABI_NAMESPACE::

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

#define DPL_HAS_CXX26_EXTENSIONS                   \
    (((DPL_COMPILER_CLANG_AT_LEAST(18, 0, 0)) |    \
         (DPL_COMPILER_GCC_AT_LEAST(14, 0, 0))) && \
        !DPL_CXX26)

#if !DPL_IS_STD && DPL_CXX
// TODO deal with std module
DPL_STD_NAMESPACE_BEGIN

template <typename>
struct tuple_size;

template <decltype(sizeof(0)), typename>
struct tuple_element;

template <typename T>
inline constexpr decltype(sizeof(0)) tuple_size_v = tuple_size<T>::value;

template <decltype(sizeof(0)) N, typename T>
using tuple_element_t = typename tuple_element<N, T>::type;

DPL_STD_NAMESPACE_END

#endif

#endif
