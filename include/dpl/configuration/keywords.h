/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_compiler.h"

#if DPL_COMPILER_MSVC
#  define DPL_HAS_KEYWORD(NAME)                            \
      DPL_IS_EMPTY(DPL_CONCAT(__DPL_KEYWORD_MSVC, NAME)) | \
          DPL_IS_EMPTY(DPL_CONCAT(__DPL_SAL_ANNOTATION, NAME))

#  define DPL_HAS_SAL_ANNOTATION(NAME) DPL_IS_EMPTY(DPL_CONCAT(__DPL_SAL_ANNOTATION, NAME))

#  define DPL_CDECL __cdecl
#  define DPL_FASTCALL __fastcall
#  define DPL_STDCALL __stdcall
#  define DPL_RESTRICT __restrict
#  define DPL_UNALIGNED __unaligned
#  define DPL_VECTORCALL __vectorcall
#  define DPL_W64 __w64
#  define DPL_FORCEINLINE __forceinline

#  define __DPL_KEYWORD_MSVC__declspec
#  define __DPL_KEYWORD_MSVC__cdecl
#  define __DPL_KEYWORD_MSVC__fastcall
#  define __DPL_KEYWORD_MSVC__stdcall
#  define __DPL_KEYWORD_MSVC__restrict
#  define __DPL_KEYWORD_MSVC__vectorcall
#  define __DPL_KEYWORD_MSVC__unaligned
#  define __DPL_KEYWORD_MSVC__w64
#  define __DPL_KEYWORD_MSVC__forceinline

/* TODO wtf */
#  define __DPL_SAL_ANNOTATION_Check_return_

#elif DPL_COMPILER_GNU_BASED
#  define DPL_HAS_KEYWORD(NAME) DPL_IS_EMPTY(DPL_CONCAT(__DPL_GNU_KEYWORD, NAME))

#  define __DPL_KEYWORD_GNU__restrict__

#  define DPL_CDECL
#  define DPL_FASTCALL
#  define DPL_STDCALL
#  define DPL_RESTRICT __restrict__
#  define DPL_UNALIGNED
#  define DPL_VECTORCALL
#  define DPL_W64
#  define DPL_FORCEINLINE
#else

#  define DPL_HAS_KEYWORD(NAME) 0
#  define DPL_CDECL
#  define DPL_FASTCALL
#  define DPL_STDCALL
#  define DPL_RESTRICT
#  define DPL_UNALIGNED
#  define DPL_VECTORCALL
#  define DPL_W64
#  define DPL_FORCEINLINE
#endif
