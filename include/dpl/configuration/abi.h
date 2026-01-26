/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/configuration/attributes.h"
#include "dpl/configuration/compiler.h" // IWYU pragma: keep
#include "dpl/configuration/exceptions.h"
#include "dpl/configuration/standard.h"
#include "dpl/configuration/target.h"       // IWYU pragma: keep
#include "dpl/preprocessor/concatenation.h" // IWYU pragma: keep
#include "dpl/preprocessor/to_string.h"

#ifndef DPL_DEFAULT_ABI_PREFIX
#  define DPL_DEFAULT_ABI_PREFIX DPL_TO_STRING(DPL)
#  if DPL_CXX
/* Ensure that the default tag is a string */
static_assert(true, DPL_DEFAULT_ABI_PREFIX);
#  endif
#endif

#if DPL_COMPILER_MSVC

#  if !DPL_HAS_DECLSPEC(dllexport) || !DPL_HAS_DECLSPEC(dllimport)
#    error "Unrecognized compiler"
#  endif

#  ifdef DPL_BUILDING_LIBRARY
#    define __DPL_ATTRIBUTE__ABI_PUBLIC (DLLEXPORT)
#    define __DPL_ATTRIBUTE_TYPE_AGGREGATE__ABI_PUBLIC
#    define __DPL_ABI_PUBLIC __declspec(dllexport)
#  else
#    define __DPL_ATTRIBUTE__ABI_PUBLIC (DLLIMPORT)
#    define __DPL_ATTRIBUTE_TYPE_AGGREGATE__ABI_PUBLIC
#    define __DPL_ABI_PUBLIC __declspec(dllimport)
#  endif
#  define __DPL_ABI_PRIVATE
#  define __DPL_ATTRIBUTE__ABI_PRIVATE
#  ifdef __DPL_ATTRIBUTE_TYPE_AGGREGATE__ABI_PRIVATE
#    error '__DPL_ATTRIBUTE_TYPE_AGGREGATE__ABI_PRIVATE' cannot be defined
#  endif
#  define __DPL_PUBLIC_TEMPLATE_DATA
#else

#  if !DPL_HAS_GNU_ATTRIBUTE(__visibility__)
#    error "Unrecognized compiler"
#  endif

#  define __DPL_ATTRIBUTE__ABI_PUBLIC (VISIBILITY("default"))
#  define __DPL_ATTRIBUTE_TYPE_AGGREGATE__ABI_PUBLIC
#  define __DPL_ATTRIBUTE__ABI_PRIVATE (VISIBILITY("hidden"))
#  define __DPL_ATTRIBUTE_TYPE_AGGREGATE__ABI_PRIVATE

#  define __DPL_ABI_PUBLIC __attribute__((__visibility__("default")))
#  define __DPL_ABI_PRIVATE __attribute__((__visibility__("hidden")))
#  define __DPL_PUBLIC_TEMPLATE_DATA __attribute__((__visibility__("default")))

#endif

#if !DPL_HAS_GNU_ATTRIBUTE(TYPE_VISIBILITY)
/* For GNU compilers that don't have type visibility we must keep the templates
 * visible */
#  define __DPL_PUBLIC_TEMPLATE __attribute__((__visibility__("default")))
#  define __DPL_ATTRIBUTE__PUBLIC_TEMPLATE (VISIBILITY("default"))
#  define __DPL_ATTRIBUTE_TYPE_AGGREGATE__PUBLIC_TEMPLATE
#  define __DPL_PUBLIC_TYPE_VISIBILITY
#  define __DPL_PRIVATE_TYPE_VISIBILITY
#else
#  define __DPL_PUBLIC_TYPE_VISIBILITY \
      __attribute__((__type_visibility__("default")))
#  define __DPL_PRIVATE_TYPE_VISIBILITY \
      __attribute__((__type_visibility__("hidden")))
#  define __DPL_PUBLIC_TEMPLATE
#  define __DPL_ATTRIBUTE__PUBLIC_TEMPLATE
#  ifdef __DPL_ATTRIBUTE_TYPE_AGGREGATE__PUBLIC_TEMPLATE
#    error '__DPL_ATTRIBUTE_TYPE_AGGREGATE__PUBLIC_TEMPLATE' cannot be defined
#  endif
#endif

#define __DPL_ABI_TAG_DELIMITER "_"
#define __DPL_ODR_SIGNATURE_1(_0, _1, _2) _0##_1##_2
#define __DPL_ODR_SIGNATURE_0(_0, _1, _2) _0 _1 _2
#define __DPL_ODR_SIGNATURE                                        \
    DPL_DEFAULT_ABI_PREFIX "_" DPL_TO_STRING(__DPL_HARDENING_MODE) \
        DPL_TO_STRING(__DPL_ABI_EXCEPTION_TAG)

#if DPL_HAS_GNU_ATTRIBUTE(__exclude_from_explicit_instantiation__)
#  define __DPL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION \
      __attribute__((__exclude_from_explicit_instantiation__))
#else
#  define __DPL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#endif

#if DPL_HAS_GNU_ATTRIBUTE(__abi_tag__)
#  define __DPL_ABI_TAG(TAG) __attribute__((__abi_tag__(TAG)))
#else
#  define __DPL_ABI_TAG(TAG)
#endif

#define __DPL_HIDE_FROM_ABI                                                    \
    __DPL_ABI_PRIVATE __DPL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION __DPL_ABI_TAG( \
        __DPL_ODR_SIGNATURE)
#define __DPL_ATTRIBUTE__HIDE_FROM_ABI                                \
    __DPL_ATTRIBUTE__ABI_PRIVATE(EXCLUDE_FROM_EXPLICIT_INSTANTIATION) \
    (ABI_TAG(__DPL_ODR_SIGNATURE))
#define __DPL_ATTRIBUTE_TYPE_AGGREGATE__HIDE_FROM_ABI

/* virtual functions must be linked to the same symbol */
#define __DPL_HIDE_FROM_ABI_VIRTUAL \
    __DPL_ABI_PRIVATE __DPL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#define __DPL_ATTRIBUTE__HIDE_FROM_ABI_VIRTUAL \
    __DPL_ATTRIBUTE__ABI_PRIVATE(EXCLUDE_FROM_EXPLICIT_INSTANTIATION)
#define __DPL_ATTRIBUTE_TYPE_AGGREGATE__HIDE_FROM_ABI_VIRTUAL
#define __DPL_HIDE_FROM_ABI_UNTAGGED \
    __DPL_ABI_PRIVATE __DPL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#define __DPL_ATTRIBUTE__HIDE_FROM_ABI_UNTAGGED \
    __DPL_ATTRIBUTE__ABI_PRIVATE(EXCLUDE_FROM_EXPLICIT_INSTANTIATION)
#define __DPL_ATTRIBUTE_TYPE_AGGREGATE__HIDE_FROM_ABI_UNTAGGED

#if DPL_WITH_EXCEPTIONS
#  define __DPL_ABI_EXCEPTION_TAG e
#else
#  define __DPL_ABI_EXCEPTION_TAG n
#endif

#ifndef __DPL_ABI_EXCEPTION_TAG
#  error Undefined exception tag
#endif

/* Currently no other hardening mode */
#define __DPL_HARDENING_MODE n

#if DPL_CXX
#  define DPL_EXTERN_C extern "C"
#  define DPL_EXTERN_C_BEGIN DPL_EXTERN_C {
#  define DPL_EXTERN_C_END }
#else
#  define DPL_EXTERN_C
#  define DPL_EXTERN_C_BEGIN
#  define DPL_EXTERN_C_END
#endif

#if DPL_COMPILER_MSVC && DPL_CXX20
#  define DPL_EXTERN_CXX extern "C++"
#  define DPL_EXTERN_CXX_BEGIN DPL_EXTERN_CXX {
#  define DPL_EXTERN_CXX_END }
#else
#  define DPL_EXTERN_CXX
#  define DPL_EXTERN_CXX_BEGIN
#  define DPL_EXTERN_CXX_END
#endif
