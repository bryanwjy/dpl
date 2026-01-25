// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/rebind_simd.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/type_traits/make_signed.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <simd_element T>
using bit_type_for DPL_NODEBUG = bit_type<sizeof(T) * char_bit_v>;
template <simd_element T>
using bit_type_for_t DPL_NODEBUG = bit_type_t<sizeof(T) * char_bit_v>;
template <simd_element T>
using sbit_type_for_t DPL_NODEBUG = make_signed_t<bit_type_for_t<T>>;

template <simd_element E, simd_abi A>
using simd_bits_t DPL_NODEBUG = basic_simd<bit_type_for_t<E>, A>;
template <simd_element E, simd_abi A>
using simd_sbits_t DPL_NODEBUG = basic_simd<bit_type_for_t<E>, A>;
template <simd_class T, simd_abi A = typename T::abi_type>
using simd_bits_for_t DPL_NODEBUG =
    rebind_simd_t<T, bit_type_for_t<simd_element_type_t<T>>, A>;
template <simd_class T, simd_abi A = typename T::abi_type>
using simd_sbits_for_t DPL_NODEBUG =
    rebind_simd_t<T, sbit_type_for_t<simd_element_type_t<T>>, A>;

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
