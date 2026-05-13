// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcast.h"
#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/initialize.h"
#include "dpl/core/basic/reinterpret.h"
#include "dpl/core/basic/to_simd_mask.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element_for.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/different_from.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename E, simd_abi A>
requires simd_element_for<E, A>
class basic_simd_mask<E, A> {
    using traits DPL_NODEBUG = simd_abi_traits<E, A>;
    using mask_type DPL_NODEBUG = typename traits::native_mask;
    using vector_type DPL_NODEBUG = typename traits::native_type;

public:
    using simd_type = basic_simd<E, A>;
    using value_type = bool;
    using abi_type = A;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t size() noexcept {
        if constexpr (fixed_width_abi<A>) {
            return traits::size;
        } else {
            return traits::size();
        }
    }

    __DPL_HIDE_FROM_ABI constexpr basic_simd_mask() noexcept
        : basic_simd_mask(datapar::broadcast<basic_simd_mask>(false)) {}

    __DPL_HIDE_FROM_ABI constexpr basic_simd_mask(mask_type data) noexcept
        : mask_{data} {}

    template <core_convertible_to<bool>... Bs>
    requires fixed_width_abi<A> && (sizeof...(Bs) == traits::size)
    __DPL_HIDE_FROM_ABI constexpr basic_simd_mask(Bs&&... args) noexcept
        : basic_simd_mask(datapar::initialize<A>(
              static_cast<bool>(__DPL forward<Bs>(args))...)) {}

    template <different_from<E> E2>
    requires simd_element<E2> && common_bits_with<E, E2>
    __DPL_HIDE_FROM_ABI constexpr basic_simd_mask(
        basic_simd_mask<E2, A> other) noexcept
        : basic_simd_mask(datapar::reinterpret<basic_simd_mask>(other)) {}

    template <common_bits_simd_with<simd_type> T>
    __DPL_HIDE_FROM_ABI explicit constexpr basic_simd_mask(
        assume_normalized_mask_t tag, T simd) noexcept
        : basic_simd_mask(datapar::to_simd_mask(
              tag, datapar::reinterpret<simd_type>(simd))) {}

    template <different_from<basic_simd_mask> T>
    requires common_bits_simd_with<T, simd_type>
    __DPL_HIDE_FROM_ABI explicit constexpr basic_simd_mask(T simd) noexcept
        : basic_simd_mask(
              datapar::to_simd_mask(datapar::reinterpret<simd_type>(simd))) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr mask_type operator+(this basic_simd_mask self) noexcept {
        return self.mask_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr bool operator[](this basic_simd_mask self,
        internal::extraction_index auto idx) noexcept {
        // assert(idx < simd_abi_traits<basic_simd_mask>::size);
        return datapar::extract(self, idx);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit constexpr operator mask_type(this basic_simd_mask self) noexcept {
        return self.mask_;
    }

private:
    mask_type mask_;
};

template <simd_element E, simd_abi A>
explicit basic_simd_mask(basic_simd<E, A>) -> basic_simd_mask<E, A>;
template <simd_element E, simd_abi A>
explicit basic_simd_mask(assume_normalized_mask_t, basic_simd<E, A>)
    -> basic_simd_mask<E, A>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
