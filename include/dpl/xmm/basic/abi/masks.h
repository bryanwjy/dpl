// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/basic/abi.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi/concepts.h"
#  include "dpl/xmm/basic/abi/type_traits.h"
#  include "dpl/xmm/basic/abi/xmmdef.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/const_mask.h"
#    include "dpl/core/type_traits/simd_abi_traits.h"
#    include "dpl/core/type_traits/unsigned_integral_type.h"
#    include "dpl/std/utility/bitset.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

// imask_t will always be integral
template <simd_element E>
using imask_t DPL_NODEBUG =
    typename bitset<simd_abi_traits<abi_tag, E>::size>::underlying_type;
template <simd_element E>
using bitset_t DPL_NODEBUG = bitset<simd_abi_traits<abi_tag, E>::size>;
template <simd_element E, imask_t<E> V>
using cmask_t DPL_NODEBUG =
    dx::const_mask<simd_abi_traits<abi_tag, E>::size, V>;
} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
