// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcast.h"
#include "dpl/core/basic/broadcastable_base.h"
#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/initialize.h"
#include "dpl/core/basic/reinterpret.h"
#include "dpl/core/basic/to_simd_mask.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/type_traits/element_count.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/type_traits/is_enum.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <simd_element E, simd_abi A>
class simd_mask_not;

DPL_EXPORT template <simd_element E, simd_abi A>
class simd_mask<E, A> {
    using mask_type DPL_NODEBUG = typename A::template native_mask<E>;
    using vector_type DPL_NODEBUG = typename A::template native_type<E>;

public:
    using simd_type = basic_simd<E, A>;
    using value_type = bool;
    using abi_type = A;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t size() noexcept { return element_count<E, A>; }

    __DPL_HIDE_FROM_ABI constexpr simd_mask() noexcept
        : simd_mask(datapar::broadcast<simd_mask>(false)) {}

    __DPL_HIDE_FROM_ABI constexpr simd_mask(mask_type data) noexcept
        : mask_{data} {}

    template <core_convertible_to<bool>... Bs>
    requires (sizeof...(Bs) == element_count<E, A>)
    __DPL_HIDE_FROM_ABI constexpr simd_mask(Bs&&... args) noexcept
        : simd_mask(datapar::initialize<A>(
              static_cast<bool>(__DPL forward<Bs>(args))...)) {}

    template <different_from<E> E2>
    requires simd_element<E2> && common_bits_with<E, E2>
    __DPL_HIDE_FROM_ABI constexpr simd_mask(simd_mask<E2, A> other) noexcept
        : simd_mask(datapar::reinterpret<simd_mask>(other)) {}

    template <common_bits_simd_with<simd_type> T>
    __DPL_HIDE_FROM_ABI explicit constexpr simd_mask(
        direct_t tag, T simd) noexcept
        : simd_mask(datapar::to_simd_mask(
              tag, datapar::reinterpret<simd_type>(simd))) {}

    template <common_bits_simd_with<simd_type> T>
    __DPL_HIDE_FROM_ABI explicit constexpr simd_mask(T simd) noexcept
        : simd_mask(
              datapar::to_simd_mask(datapar::reinterpret<simd_type>(simd))) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr mask_type operator+(this simd_mask self) noexcept {
        return self.mask_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr bool operator[](
        this simd_mask self, internal::extraction_index auto idx) noexcept {
        assert(idx < element_count<simd_mask>);
        return datapar::extract(self, idx);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr simd_mask_not<E, A> operator!(this simd_mask self) noexcept {
        return self.mask_;
    }

private:
    mask_type mask_;
};

template <simd_element E, simd_abi A>
explicit simd_mask(basic_simd<E, A>) -> simd_mask<E, A>;
template <simd_element E, simd_abi A>
explicit simd_mask(direct_t, basic_simd<E, A>) -> simd_mask<E, A>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
