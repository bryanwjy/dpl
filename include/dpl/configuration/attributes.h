/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_builtins.h"
#include "configuration/utl_compiler.h"
#include "configuration/utl_declspec.h"
#include "configuration/utl_keywords.h"

#if defined(__cplusplus) && defined(__has_cpp_attribute)
#  define DPL_HAS_CPP_ATTRIBUTE(NAME) __has_cpp_attribute(NAME)
#else /* ifdef __has_cpp_attribute */
#  define DPL_HAS_CPP_ATTRIBUTE(NAME) 0
#endif /* ifdef __has_cpp_attribute */

#ifdef __has_attribute
#  define DPL_HAS_GNU_ATTRIBUTE(NAME) __has_attribute(NAME)
#else /* ifdef __has_attribute */
#  define DPL_HAS_GNU_ATTRIBUTE(NAME) 0
#endif /* ifdef __has_attribute */

#if DPL_HAS_CPP_ATTRIBUTE(nodiscard) && DPL_CXX17
#  define DPL_NODISCARD [[nodiscard]]
#  define __DPL_ATTRIBUTE_NODISCARD nodiscard
#  define __DPL_ATTRIBUTE_TYPE_CPP_NODISCARD
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::warn_unused_result)
#  define DPL_NODISCARD [[gnu::warn_unused_result]]
#  define __DPL_ATTRIBUTE_NODISCARD gnu::warn_unused_result
#  define __DPL_ATTRIBUTE_TYPE_CPP_NODISCARD
#elif DPL_HAS_CPP_ATTRIBUTE(clang::warn_unused_result)
#  define DPL_NODISCARD [[clang::warn_unused_result]]
#  define __DPL_ATTRIBUTE_NODISCARD clang::warn_unused_result
#  define __DPL_ATTRIBUTE_TYPE_CPP_NODISCARD
#elif DPL_HAS_GNU_ATTRIBUTE(__warn_unused_result__)
#  define DPL_NODISCARD __attribute__((__warn_unused_result__))
#  define __DPL_ATTRIBUTE_NODISCARD __warn_unused_result__
#  define __DPL_ATTRIBUTE_TYPE_GNU_NODISCARD
#elif DPL_HAS_KEYWORD(_Check_return_)
#  define DPL_NODISCARD _Check_return_
#  define __DPL_ATTRIBUTE_NODISCARD _Check_return_
#  define __DPL_ATTRIBUTE_TYPE_MSVC_EXT_NODISCARD
#else
#  define DPL_NODISCARD
#endif /* DPL_HAS_CPP_ATTRIBUTE(nodiscard) */

#if DPL_HAS_CPP_ATTRIBUTE(noreturn)
#  define DPL_NORETURN [[noreturn]]
#  define __DPL_ATTRIBUTE_NORETURN noreturn
#  define __DPL_ATTRIBUTE_TYPE_CPP_NORETURN
#else
#  define DPL_NORETURN
#endif

#if DPL_HAS_CPP_ATTRIBUTE(msvc::lifetimebound)
#  define DPL_LIFETIMEBOUND [[msvc::lifetimebound]]
#  define __DPL_ATTRIBUTE_LIFETIMEBOUND msvc::lifetimebound
#  define __DPL_ATTRIBUTE_TYPE_CPP_LIFETIMEBOUND
#elif DPL_HAS_CPP_ATTRIBUTE(clang::lifetimebound) /* DPL_HAS_CPP_ATTRIBUTE(msvc::lifetimebound) */
#  define DPL_LIFETIMEBOUND [[clang::lifetimebound]]
#  define __DPL_ATTRIBUTE_LIFETIMEBOUND clang::lifetimebound
#  define __DPL_ATTRIBUTE_TYPE_CPP_LIFETIMEBOUND
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::lifetimebound) /* DPL_HAS_CPP_ATTRIBUTE(msvc::lifetimebound) */
#  define DPL_LIFETIMEBOUND [[gnu::lifetimebound]]
#  define __DPL_ATTRIBUTE_LIFETIMEBOUND gnu::lifetimebound
#  define __DPL_ATTRIBUTE_TYPE_CPP_LIFETIMEBOUND
#elif DPL_HAS_GNU_ATTRIBUTE(__lifetimebound__) /* DPL_HAS_CPP_ATTRIBUTE(msvc::lifetimebound) */
#  define DPL_LIFETIMEBOUND __attribute__((__lifetimebound__))
#  define __DPL_ATTRIBUTE_LIFETIMEBOUND __lifetimebound__
#  define __DPL_ATTRIBUTE_TYPE_GNU_LIFETIMEBOUND
#else
#  define DPL_LIFETIMEBOUND
#endif /* DPL_HAS_CPP_ATTRIBUTE(msvc::lifetimebound) */

#if DPL_HAS_CPP_ATTRIBUTE(gnu::pure)
#  define __DPL_ATTRIBUTE_PURE gnu::pure
#  define __DPL_ATTRIBUTE_TYPE_CPP_PURE
#elif DPL_HAS_GNU_ATTRIBUTE(__pure__) /* DPL_HAS_CPP_ATTRIBUTE(gnu::pure) */
#  define __DPL_ATTRIBUTE_PURE __pure__
#  define __DPL_ATTRIBUTE_TYPE_GNU_PURE
#endif /* DPL_HAS_CPP_ATTRIBUTE(gnu::pure) */

#if DPL_HAS_CPP_ATTRIBUTE(gnu::const)
#  define __DPL_ATTRIBUTE_CONST gnu::const
#  define __DPL_ATTRIBUTE_TYPE_CPP_CONST
#elif DPL_HAS_GNU_ATTRIBUTE(__const__)
#  define __DPL_ATTRIBUTE_CONST __const__
#  define __DPL_ATTRIBUTE_TYPE_GNU_CONST
#elif DPL_HAS_DECLSPEC(noalias)
#  define __DPL_ATTRIBUTE_CONST noalias
#  define __DPL_ATTRIBUTE_TYPE_DECLSPEC_CONST
#endif /* DPL_HAS_CPP_ATTRIBUTE(gnu::const) */

#if DPL_HAS_CPP_ATTRIBUTE(msvc::intrinsic)
#  define __DPL_ATTRIBUTE_INTRINSIC msvc::intrinsic
#  define __DPL_ATTRIBUTE_TYPE_CPP_INTRINSIC
#endif /* DPL_HAS_CPP_ATTRIBUTE(msvc::intrinsic) */

#if DPL_HAS_CPP_ATTRIBUTE(gnu::flatten)
#  define __DPL_ATTRIBUTE_FLATTEN gnu::flatten
#  define __DPL_ATTRIBUTE_TYPE_CPP_FLATTEN
#elif DPL_HAS_CPP_ATTRIBUTE(msvc::flatten)
#  define __DPL_ATTRIBUTE_FLATTEN msvc::flatten
#  define __DPL_ATTRIBUTE_TYPE_CPP_FLATTEN
#elif DPL_HAS_GNU_ATTRIBUTE(__flatten__)
#  define __DPL_ATTRIBUTE_FLATTEN __flatten__
#  define __DPL_ATTRIBUTE_TYPE_GNU_FLATTEN
#endif /* DPL_HAS_CPP_ATTRIBUTE(gnu::const) */

#if DPL_HAS_CPP_ATTRIBUTE(msvc::no_unique_address)
#  define __DPL_ATTRIBUTE_NO_UNIQUE_ADDRESS msvc::no_unique_address
#  define __DPL_ATTRIBUTE_TYPE_CPP_NO_UNIQUE_ADDRESS
#elif DPL_HAS_CPP_ATTRIBUTE(no_unique_address)
#  define __DPL_ATTRIBUTE_NO_UNIQUE_ADDRESS no_unique_address
#  define __DPL_ATTRIBUTE_TYPE_CPP_NO_UNIQUE_ADDRESS
#elif DPL_HAS_GNU_ATTRIBUTE(__no_unique_address__)
#  define __DPL_ATTRIBUTE_NO_UNIQUE_ADDRESS __no_unique_address__
#  define __DPL_ATTRIBUTE_TYPE_GNU_NO_UNIQUE_ADDRESS
#endif /* DPL_HAS_CPP_ATTRIBUTE(msvc::no_unique_address) */

#if DPL_HAS_CPP_ATTRIBUTE(clang::always_inline)
#  define __DPL_ATTRIBUTE_ALWAYS_INLINE clang::always_inline
#  define __DPL_ATTRIBUTE_TYPE_CPP_ALWAYS_INLINE
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::always_inline)
#  define __DPL_ATTRIBUTE_ALWAYS_INLINE gnu::always_inline
#  define __DPL_ATTRIBUTE_TYPE_CPP_ALWAYS_INLINE
#elif DPL_HAS_CPP_ATTRIBUTE(msvc::forceinline)
#  define __DPL_ATTRIBUTE_ALWAYS_INLINE msvc::forceinline
#  define __DPL_ATTRIBUTE_TYPE_CPP_ALWAYS_INLINE
#elif DPL_HAS_GNU_ATTRIBUTE(__always_inline__)
#  define __DPL_ATTRIBUTE_ALWAYS_INLINE __always_inline__
#  define __DPL_ATTRIBUTE_TYPE_GNU_ALWAYS_INLINE
#elif DPL_HAS_MSVC_KEYWORD(__forceinline)
#  define __DPL_ATTRIBUTE_ALWAYS_INLINE __forceinline
#  define __DPL_ATTRIBUTE_TYPE_MSVC_EXT_ALWAYS_INLINE
#endif /* DPL_HAS_CPP_ATTRIBUTE(clang::always_inline) */

#if DPL_HAS_CPP_ATTRIBUTE(msvc::noinline)
#  define __DPL_ATTRIBUTE_NOINLINE msvc::noinline
#  define __DPL_ATTRIBUTE_TYPE_CPP_NOINLINE
#elif DPL_HAS_CPP_ATTRIBUTE(clang::noinline)
#  define __DPL_ATTRIBUTE_NOINLINE clang::noinline
#  define __DPL_ATTRIBUTE_TYPE_CPP_NOINLINE
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::noinline)
#  define __DPL_ATTRIBUTE_NOINLINE gnu::noinline
#  define __DPL_ATTRIBUTE_TYPE_CPP_NOINLINE
#elif DPL_HAS_GNU_ATTRIBUTE(__noinline__)
#  define __DPL_ATTRIBUTE_NOINLINE __noinline__
#  define __DPL_ATTRIBUTE_TYPE_GNU_NOINLINE
#elif DPL_HAS_DECLSPEC(noinline)
#  define __DPL_ATTRIBUTE_NOINLINE noinline
#  define __DPL_ATTRIBUTE_TYPE_DECLSPEC_NOINLINE
#endif /* DPL_HAS_CPP_ATTRIBUTE(clang::noinline) */

#if DPL_HAS_CPP_ATTRIBUTE(clang::malloc)
#  define __DPL_ATTRIBUTE_MALLOC clang::malloc
#  define __DPL_ATTRIBUTE_TYPE_CPP_MALLOC
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::malloc)
#  define __DPL_ATTRIBUTE_MALLOC gnu::malloc
#  define __DPL_ATTRIBUTE_TYPE_CPP_MALLOC
#elif DPL_HAS_GNU_ATTRIBUTE(__malloc__)
#  define __DPL_ATTRIBUTE_MALLOC __malloc__
#  define __DPL_ATTRIBUTE_TYPE_GNU_MALLOC
#elif DPL_HAS_DECLSPEC(allocator) | DPL_HAS_DECLSPEC(restrict)
#  if DPL_HAS_DECLSPEC(allocator) & DPL_HAS_DECLSPEC(restrict)
#    define __DPL_ATTRIBUTE_MALLOC allocator, restrict
#    define __DPL_ATTRIBUTE_TYPE_DECLSPEC_MALLOC
#  elif DPL_HAS_DECLSPEC(allocator)
#    define __DPL_ATTRIBUTE_MALLOC allocator
#    define __DPL_ATTRIBUTE_TYPE_DECLSPEC_MALLOC
#  elif DPL_HAS_DECLSPEC(restrict)
#    define __DPL_ATTRIBUTE_MALLOC restrict
#    define __DPL_ATTRIBUTE_TYPE_DECLSPEC_MALLOC
#  endif
#endif /* DPL_HAS_CPP_ATTRIBUTE(clang::malloc) */

#if DPL_COMPILER_GCC
#  if DPL_HAS_CPP_ATTRIBUTE(gnu::malloc)
#    define __DPL_ATTRIBUTE_DEALLOCATED_BY gnu::malloc
#    define __DPL_ATTRIBUTE_TYPE_CPP_DEALLOCATED_BY(FUNC, ...)
#  elif DPL_HAS_GNU_ATTRIBUTE(__malloc__)
#    define __DPL_ATTRIBUTE_DEALLOCATED_BY __malloc__
#    define __DPL_ATTRIBUTE_TYPE_GNU_DEALLOCATED_BY(FUNC, ...)
#  endif /* DPL_HAS_CPP_ATTRIBUTE(clang::malloc) */
#endif

#if DPL_HAS_CPP_ATTRIBUTE(visibility)
#  define __DPL_ATTRIBUTE_VISIBILITY visibility
#  define __DPL_ATTRIBUTE_TYPE_CPP_VISIBILITY(VISIBILITY)
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::visibility)
#  define __DPL_ATTRIBUTE_VISIBILITY gnu::visibility
#  define __DPL_ATTRIBUTE_TYPE_CPP_VISIBILITY(VISIBILITY)
#elif DPL_HAS_GNU_ATTRIBUTE(__visibility__)
#  define __DPL_ATTRIBUTE_VISIBILITY __visibility__
#  define __DPL_ATTRIBUTE_TYPE_GNU_VISIBILITY(VISIBILITY)
#endif

#if DPL_HAS_CPP_ATTRIBUTE(type_visibility) && DPL_CXX17
#  define __DPL_ATTRIBUTE_TYPE_VISIBILITY type_visibility
#  define __DPL_ATTRIBUTE_TYPE_CPP_TYPE_VISIBILITY(VISIBILITY)
#elif DPL_HAS_CPP_ATTRIBUTE(clang::type_visibility) && DPL_CXX17
#  define __DPL_ATTRIBUTE_TYPE_VISIBILITY clang::type_visibility
#  define __DPL_ATTRIBUTE_TYPE_CPP_TYPE_VISIBILITY(VISIBILITY)
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::type_visibility) && DPL_CXX17
#  define __DPL_ATTRIBUTE_TYPE_VISIBILITY gnu::type_visibility
#  define __DPL_ATTRIBUTE_TYPE_CPP_TYPE_VISIBILITY(VISIBILITY)
#elif DPL_HAS_GNU_ATTRIBUTE(__type_visibility__)
#  define __DPL_ATTRIBUTE_TYPE_VISIBILITY __type_visibility__
#  define __DPL_ATTRIBUTE_TYPE_GNU_TYPE_VISIBILITY(VISIBILITY)
#endif

#if DPL_HAS_CPP_ATTRIBUTE(gnu::dllimport)
#  define __DPL_ATTRIBUTE_DLLIMPORT gnu::dllimport
#  define __DPL_ATTRIBUTE_TYPE_CPP_DLLIMPORT
#elif DPL_HAS_GNU_ATTRIBUTE(__dllimport__)
#  define __DPL_ATTRIBUTE_DLLIMPORT __dllimport__
#  define __DPL_ATTRIBUTE_TYPE_CPP_DLLIMPORT
#elif DPL_HAS_DECLSPEC(dllimport)
#  define __DPL_ATTRIBUTE_DLLIMPORT dllimport
#  define __DPL_ATTRIBUTE_TYPE_DECLSPEC_DLLIMPORT
#endif

#if DPL_HAS_CPP_ATTRIBUTE(gnu::dllexport)
#  define __DPL_ATTRIBUTE_DLLEXPORT gnu::dllexport
#  define __DPL_ATTRIBUTE_TYPE_CPP_DLLEXPORT
#elif DPL_HAS_GNU_ATTRIBUTE(__dllexport__)
#  define __DPL_ATTRIBUTE_DLLEXPORT __dllexport__
#  define __DPL_ATTRIBUTE_TYPE_CPP_DLLEXPORT
#elif DPL_HAS_DECLSPEC(dllexport)
#  define __DPL_ATTRIBUTE_DLLIMPORT dllexport
#  define __DPL_ATTRIBUTE_TYPE_DECLSPEC_DLLIMPORT
#endif

#if DPL_HAS_CPP_ATTRIBUTE(clang::exclude_from_explicit_instantiation)
#  define __DPL_ATTRIBUTE_EXCLUDE_FROM_EXPLICIT_INSTANTIATION \
      clang::exclude_from_explicit_instantiation
#  define __DPL_ATTRIBUTE_TYPE_CPP_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::exclude_from_explicit_instantiation)
#  define __DPL_ATTRIBUTE_EXCLUDE_FROM_EXPLICIT_INSTANTIATION \
      gnu::exclude_from_explicit_instantiation
#  define __DPL_ATTRIBUTE_TYPE_CPP_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#elif DPL_HAS_GNU_ATTRIBUTE(__exclude_from_explicit_instantiation__)
#  define __DPL_ATTRIBUTE_EXCLUDE_FROM_EXPLICIT_INSTANTIATION \
      __exclude_from_explicit_instantiation__
#  define __DPL_ATTRIBUTE_TYPE_GNU_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#endif

#if DPL_HAS_CPP_ATTRIBUTE(clang::abi_tag)
#  define __DPL_ATTRIBUTE_ABI_TAG clang::abi_tag
#  define __DPL_ATTRIBUTE_TYPE_CPP_ABI_TAG(STR)
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::abi_tag)
#  define __DPL_ATTRIBUTE_ABI_TAG gnu::abi_tag
#  define __DPL_ATTRIBUTE_TYPE_CPP_ABI_TAG(STR)
#elif DPL_HAS_GNU_ATTRIBUTE(__abi_tag__)
#  define __DPL_ATTRIBUTE_ABI_TAG __abi_tag__
#  define __DPL_ATTRIBUTE_TYPE_GNU_ABI_TAG(STR)
#endif

#if DPL_HAS_CPP_ATTRIBUTE(clang::reinitializes)
#  define __DPL_ATTRIBUTE_REINITIALIZES clang::reinitializes
#  define __DPL_ATTRIBUTE_TYPE_CPP_REINITIALIZES
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::reinitializes)
#  define __DPL_ATTRIBUTE_REINITIALIZES gnu::reinitializes
#  define __DPL_ATTRIBUTE_TYPE_CPP_REINITIALIZES
#elif DPL_HAS_GNU_ATTRIBUTE(__reinitializes__)
#  define __DPL_ATTRIBUTE_REINITIALIZES __reinitializes__
#  define __DPL_ATTRIBUTE_TYPE_GNU_REINITIALIZES
#endif /* DPL_HAS_CPP_ATTRIBUTE(clang::reinitializes) */

#if DPL_HAS_CPP_ATTRIBUTE(maybe_unused) && DPL_CXX17
#  define __DPL_ATTRIBUTE_MAYBE_UNUSED maybe_unused
#  define __DPL_ATTRIBUTE_TYPE_CPP_MAYBE_UNUSED
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::unused)
#  define __DPL_ATTRIBUTE_MAYBE_UNUSED gnu::unused
#  define __DPL_ATTRIBUTE_TYPE_CPP_MAYBE_UNUSED
#elif DPL_HAS_GNU_ATTRIBUTE(__unused__)
#  define __DPL_ATTRIBUTE_MAYBE_UNUSED __unused__
#  define __DPL_ATTRIBUTE_TYPE_GNU_MAYBE_UNUSED
#endif /* DPL_HAS_CPP_ATTRIBUTE(maybe_unused) */

#if DPL_HAS_CPP_ATTRIBUTE(fallthrough) && DPL_CXX17
#  define DPL_FALLTHROUGH [[fallthrough]]
#  define __DPL_ATTRIBUTE_FALLTHROUGH fallthrough
#  define __DPL_ATTRIBUTE_TYPE_CPP_FALLTHROUGH
#elif DPL_HAS_CPP_ATTRIBUTE(gnu::fallthrough)
#  define DPL_FALLTHROUGH [[gnu::fallthrough]]
#  define __DPL_ATTRIBUTE_FALLTHROUGH gnu::fallthrough
#  define __DPL_ATTRIBUTE_TYPE_CPP_FALLTHROUGH
#elif DPL_HAS_GNU_ATTRIBUTE(__fallthrough__)
#  define DPL_FALLTHROUGH __attribute__((__fallthrough__))
#  define __DPL_ATTRIBUTE_FALLTHROUGH __fallthrough__
#  define __DPL_ATTRIBUTE_TYPE_GNU_FALLTHROUGH
#else
#  define DPL_FALLTHROUGH
#endif

#if DPL_HAS_CPP_ATTRIBUTE(clang::noescape)
#  define __DPL_ATTRIBUTE_NOESCAPE clang::noescape
#  define __DPL_ATTRIBUTE_TYPE_CPP_NOESCAPE
#endif

#if DPL_HAS_GNU_ATTRIBUTE(__optimize__)
#  define __DPL_ATTRIBUTE_OPTIMIZE __optimize__
#  define __DPL_ATTRIBUTE_TYPE_GNU_OPTIMIZE(STR)
#endif

#if DPL_HAS_CPP_ATTRIBUTE(clang::nodebug)
#  define DPL_NODEBUG [[clang::nodebug]]
#  define __DPL_ATTRIBUTE_NODEBUG clang::nodebug
#  define __DPL_ATTRIBUTE_TYPE_CPP_NODEBUG
#else
#  define DPL_NODEBUG
#endif

#if DPL_HAS_CPP_ATTRIBUTE(likely) && DPL_CXX20
#  define DPL_LIKELY [[likely]]
#  define __DPL_ATTRIBUTE_LIKELY likely
#  define __DPL_ATTRIBUTE_TYPE_CPP_LIKELY
#elif DPL_HAS_CPP_ATTRIBUTE(clang::likely)
#  define DPL_LIKELY [[clang::likely]]
#  define __DPL_ATTRIBUTE_LIKELY clang::likely
#  define __DPL_ATTRIBUTE_TYPE_CPP_LIKELY
#else
#  define DPL_LIKELY
#endif

#if DPL_HAS_CPP_ATTRIBUTE(unlikely) && DPL_CXX20
#  define DPL_UNLIKELY [[unlikely]]
#  define __DPL_ATTRIBUTE_UNLIKELY unlikely
#  define __DPL_ATTRIBUTE_TYPE_CPP_UNLIKELY
#elif DPL_HAS_CPP_ATTRIBUTE(clang::unlikely)
#  define DPL_UNLIKELY [[clang::unlikely]]
#  define __DPL_ATTRIBUTE_UNLIKELY clang::unlikely
#  define __DPL_ATTRIBUTE_TYPE_CPP_UNLIKELY
#else
#  define DPL_UNLIKELY
#endif

#if DPL_HAS_CPP_ATTRIBUTE(clang::vectorcall)
#  define __DPL_ATTRIBUTE_VECTORCALL clang::vectorcall
#  define __DPL_ATTRIBUTE_TYPE_CPP_VECTORCALL
#elif DPL_HAS_KEYWORD(__vectorcall)
#  define DPL_NODISCARD __vectorcall
#  define __DPL_ATTRIBUTE_VECTORCALL __vectorcall
#  define __DPL_ATTRIBUTE_TYPE_MSVC_EXT_VECTORCALL
#endif /* DPL_HAS_CPP_ATTRIBUTE(clang::vectorcall) */
