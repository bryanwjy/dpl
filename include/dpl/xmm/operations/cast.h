

// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/constants/value_bits.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/reinterpret.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <common_float_with<float> TE, brain_float FE>
inline simd<TE> cast(abi_tag, simd<FE> from) noexcept {
#if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtpbh_ps(+from);
#else
    auto const src = __DPL bit_cast<__m128i>(+from);
    return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(src), 16));
#endif
}

template <brain_float TE, common_float_with<float> FE>
inline simd<TE> cast(abi_tag, simd<FE> from) noexcept {
#if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtneps_pbh(+from);
#else
    auto const ival = _mm_castps_si128(+from);
    // round to nearest even
    auto const lsb = _mm_and_si128(_mm_srli_epi32(ival, 16), _mm_set1_epi32(1));
    auto const bias = _mm_add_epi32(_mm_set1_epi32(0x7fff), lsb);
    return __DPL bit_cast<__m128bh>(
        _mm_srli_epi32(_mm_add_epi32(ival, bias), 16));
#endif
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
