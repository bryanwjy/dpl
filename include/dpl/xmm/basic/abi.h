// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

// IWYU pragma: begin_exports
#  include "dpl/xmm/basic/abi/concepts.h"
#  include "dpl/xmm/basic/abi/iota.h"
#  include "dpl/xmm/basic/abi/masks.h"
#  include "dpl/xmm/basic/abi/type_traits.h"
#  include "dpl/xmm/basic/abi/xmmdef.h"
// IWYU pragma: end_exports

#  if !DPL_MODULES
#    include "dpl/core/type_traits/enable_simd_abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

struct abi_tag : simd_abi_base<abi_tag> {

    static constexpr size_t size = 16;
    static constexpr size_t alignment = 16;

    template <typename E>
    requires requires {
        typename native_vector_t<E>;
        requires native_vector_type<typename xmm::native_vector_t<E>>;
    }
    using native_vector = typename xmm::native_vector_t<E>;
    template <typename E>
    requires requires { typename native_vector<E>; }
    using native_mask = native_vector<E>;
};

inline constexpr abi_tag abi{};

using imask16_t DPL_NODEBUG = imask_t<int8>;
using imask8_t DPL_NODEBUG = imask_t<int16>;
using imask4_t DPL_NODEBUG = imask_t<int32>;
using imask2_t DPL_NODEBUG = imask_t<int64>;

template <imask16_t V>
using cmask16_t DPL_NODEBUG = dx::const_mask<16, V>;
template <imask8_t V>
using cmask8_t DPL_NODEBUG = dx::const_mask<8, V>;
template <imask4_t V>
using cmask4_t DPL_NODEBUG = dx::const_mask<4, V>;
template <imask2_t V>
using cmask2_t DPL_NODEBUG = dx::const_mask<2, V>;

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
