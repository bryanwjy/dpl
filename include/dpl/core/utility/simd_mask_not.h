// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/basic/extract.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/basic/simd_mask.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/logic.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/element_count.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace internal {
template <typename T>
inline constexpr bool is_mask_not_specialization = false;
template <simd_element E, simd_abi A>
inline constexpr bool is_mask_not_specialization<simd_mask_not<E, A>> = true;
} // namespace internal

DPL_EXPORT template <simd_element E, simd_abi A>
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

    template <common_size_simd_with<simd_mask<E, A>> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit operator T(this simd_mask_not self) noexcept {
        return datapar::bwnot(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr mask_type operator+(this simd_mask_not self) noexcept {
        return +datapar::bwnot(!self);
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
    friend constexpr auto select(common_abi_with<abi_type> auto,
        simd_mask_not self, T lhs, F rhs) noexcept {
        return datapar::select(!self, rhs, lhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool any_of(
        common_abi_with<abi_type> auto, simd_mask_not self) noexcept {
        return !datapar::none_of(internal::abi<A>, !self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool all_of(
        common_abi_with<abi_type> auto, simd_mask_not self) noexcept {
        return datapar::none_of(internal::abi<A>, !self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool none_of(
        common_abi_with<abi_type> auto, simd_mask_not self) noexcept {
        return datapar::all_of(internal::abi<A>, !self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool some_of(
        common_abi_with<abi_type> auto, simd_mask_not self) noexcept {
        return datapar::some_of(internal::abi<A>, !self);
    }

    template <typename T, typename F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto select(common_abi_with<abi_type> auto,
        simd_mask_not self, T lhs, F rhs) noexcept
    requires requires { datapar::select(!self, rhs, lhs); }
    {
        return datapar::select(!self, rhs, lhs);
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwand(common_abi_with<abi_type> auto,
        simd_mask_not self, simd_mask_not<E2, A2> arg) noexcept
    requires requires { !datapar::bwor(!self, !arg); }
    {
        return !datapar::bwor(!self, !arg);
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwnot(
        common_abi_with<abi_type> auto, simd_mask_not self) noexcept {
        return !self;
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwor(common_abi_with<abi_type> auto,
        simd_mask_not self, simd_mask_not<E2, A2> arg) noexcept
    requires requires { !datapar::bwand(!self, !arg); }
    {
        return !datapar::bwand(!self, !arg);
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwxor(common_abi_with<abi_type> auto,
        simd_mask_not self, simd_mask_not<E2, A2> arg) noexcept
    requires requires { datapar::bwxor(!self, !arg); }
    {
        return datapar::bwxor(!self, !arg);
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwandnot(common_abi_with<abi_type> auto,
        simd_mask_not self, simd_mask_not<E2, A2> arg) noexcept
    requires requires { datapar::bwandnot(!arg, !self); }
    {
        return datapar::bwandnot(!arg, !self);
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwornot(common_abi_with<abi_type> auto,
        simd_mask_not self, simd_mask_not<E2, A2> arg) noexcept
    requires requires { datapar::bwornot(!arg, !self); }
    {
        return datapar::bwornot(!arg, !self);
    }

    template <simd_mask_type M>
    requires (!internal::is_mask_not_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwand(
        common_abi_with<abi_type> auto, simd_mask_not self, M arg) noexcept
    requires requires { datapar::bwandnot(arg, !self); }
    {
        return datapar::bwandnot(arg, !self);
    }

    template <simd_mask_type M>
    requires (!internal::is_mask_not_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwor(
        common_abi_with<abi_type> auto, simd_mask_not self, M arg) noexcept
    requires requires { datapar::bwornot(arg, !self); }
    {
        return datapar::bwornot(arg, !self);
    }

    template <simd_mask_type M>
    requires (!internal::is_mask_not_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwxor(
        common_abi_with<abi_type> auto, simd_mask_not self, M arg) noexcept
    requires requires { !datapar::bwxor(!self, arg); }
    {
        return !datapar::bwxor(!self, arg);
    }

    template <simd_mask_type M>
    requires (!internal::is_mask_not_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwandnot(
        common_abi_with<abi_type> auto, simd_mask_not self, M arg) noexcept
    requires requires { !datapar::bwor(!arg, self); }
    {
        return !datapar::bwor(!arg, self);
    }

    template <simd_mask_type M>
    requires (!internal::is_mask_not_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwornot(
        common_abi_with<abi_type> auto, simd_mask_not self, M arg) noexcept
    requires requires { !datapar::bwand(!arg, self); }
    {
        return !datapar::bwand(!arg, self);
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_keep(common_abi_with<abi_type> auto,
        simd_mask_not self, simd_mask_not<E2, A2> arg) noexcept
    requires requires { !datapar::bit_fill(!self, !arg); }
    {
        return !datapar::bit_fill(!self, !arg);
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_drop(common_abi_with<abi_type> auto,
        simd_mask_not self, simd_mask_not<E2, A2> arg) noexcept
    requires requires { !datapar::bit_stencil(!self, !arg); }
    {
        return !datapar::bit_stencil(!self, !arg);
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_fill(common_abi_with<abi_type> auto,
        simd_mask_not self, simd_mask_not<E2, A2> arg) noexcept
    requires requires { !datapar::bit_keep(!self, !arg); }
    {
        return !datapar::bit_keep(!self, !arg);
    }

    template <common_size_with<E> E2, common_abi_with<abi_type> A2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_stencil(common_abi_with<abi_type> auto,
        simd_mask_not self, simd_mask_not<E2, A2> arg) noexcept
    requires requires { !datapar::bit_drop(!self, !arg); }
    {
        return !datapar::bit_drop(!self, !arg);
    }

    template <typename T>
    requires (!internal::is_mask_not_specialization<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_keep(
        common_abi_with<abi_type> auto, simd_mask_not self, T arg) noexcept
    requires requires { datapar::bit_drop(!self, arg); }
    {
        return datapar::bit_drop(!self, arg);
    }

    template <typename T>
    requires (!internal::is_mask_not_specialization<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_drop(
        common_abi_with<abi_type> auto, simd_mask_not self, T arg) noexcept
    requires requires { datapar::bit_keep(!self, arg); }
    {
        return datapar::bit_keep(!self, arg);
    }

    template <typename T>
    requires (!internal::is_mask_not_specialization<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_fill(
        common_abi_with<abi_type> auto, simd_mask_not self, T arg) noexcept
    requires requires { datapar::bit_stencil(!self, arg); }
    {
        return datapar::bit_stencil(!self, arg);
    }

    template <typename T>
    requires (!internal::is_mask_not_specialization<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_stencil(
        common_abi_with<abi_type> auto, simd_mask_not self, T arg) noexcept
    requires requires { datapar::bit_fill(!self, arg); }
    {
        return datapar::bit_fill(!self, arg);
    }

    template <typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_select(common_abi_with<abi_type> auto,
        simd_mask_not self, L lhs, R rhs) noexcept
    requires requires { datapar::bit_select(!self, rhs, lhs); }
    {
        return datapar::bit_select(!self, rhs, lhs);
    }

private:
    mask_type mask_;
};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
