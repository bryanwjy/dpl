// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/core/basic/extract.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/basic/simd_mask.h"
#  include "dpl/core/basic/to_simd_mask.h"
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
class simd_mask_not {
    using mask_type = typename A::template native_mask<E>;
    using vector_type = typename A::template native_type<E>;

public:
    using simd_type = basic_simd<E, A>;
    using value_type = bool;
    using abi_type = A;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t size() noexcept { return element_count<E, A>; }

    __DPL_HIDE_FROM_ABI constexpr simd_mask_not() noexcept
        : simd_mask_not(+simd_mask<E, A>()) {}

    template <simd_element T>
    requires (!same_as<T, E>) &&
        regular_invocable<internal::reinterpret_t<E>, simd_mask<T, A>>
    __DPL_HIDE_FROM_ABI constexpr simd_mask_not(
        simd_mask_not<T, A> other) noexcept
        : simd_mask_not(+datapar::reinterpret<E>(!other)) {}

    template <common_bits_simd_with<simd_type> T>
    requires same_as<mask_type, vector_type>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit operator T(this simd_mask_not self) noexcept {
        // return datapar::bwnot(!self);
    }

    template <common_bits_simd_with<simd_type> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit operator T(this simd_mask_not self) noexcept {
        // return datapar::bwnot(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr mask_type operator+(this simd_mask_not self) noexcept {
        // return +datapar::bwnot(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr bool operator[](
        this simd_mask_not self, internal::extraction_index auto idx) noexcept {
        assert(idx < element_count<simd_mask_not>);
        return !datapar::extract(!self, idx);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr simd_mask<E, A> operator!(this simd_mask_not self) noexcept {
        return self.mask_;
    }

    template <datapar::simd_type T, datapar::simd_type F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto select(
        abi_type, simd_mask_not self, T lhs, F rhs) noexcept {
        // return datapar::select(!self, rhs, lhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool any_of(abi_type, simd_mask_not self) noexcept {
        // return !datapar::none_of(internal::abi<A>, !self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool all_of(abi_type, simd_mask_not self) noexcept {
        // return datapar::none_of(internal::abi<A>, !self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool none_of(abi_type, simd_mask_not self) noexcept {
        // return datapar::all_of(internal::abi<A>, !self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool some_of(abi_type, simd_mask_not self) noexcept {
        // return datapar::some_of(internal::abi<A>, !self);
    }

    template <datapar::simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_set(
        abi_type, simd_mask_not self, T arg) noexcept {
        // return datapar::select(!self, arg, datapar::all_bits);
    }

    template <datapar::simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_clear(
        abi_type, simd_mask_not self, T arg) noexcept {
        // return datapar::select(!self, arg);
    }

private:
    mask_type mask_;
};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
