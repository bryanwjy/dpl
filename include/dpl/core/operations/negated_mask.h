// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bit.h"
#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/logic_reduction.h"
#include "dpl/core/operations/select.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/basic/extract.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/basic/to_native_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/concepts/simd_mask_type.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename T>
inline constexpr bool is_negated_mask_specialization = false;

template <simd_mask_type T>
class negated_mask {
    using element_type = simd_lane_type_t<T>;

public:
    using simd_type = basic_simd<element_type, typename T::abi_type>;
    using abi_type = typename T::abi_type;
    using value_type = bool;

private:
    using mask_type DPL_NODEBUG =
        typename abi_type::template native_mask<element_type>;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t size() noexcept { return simd_abi_traits<T>::size; }

    __DPL_HIDE_FROM_ABI constexpr negated_mask() noexcept
        : negated_mask(dx::to_native_type(basic_type_t<T>())) {}

    __DPL_HIDE_FROM_ABI constexpr negated_mask(T mask) noexcept : mask_(mask) {}

    template <common_size_simd_with<T> U>
    requires (!same_as<T, U>) && same_abi_simd_as<T, U> &&
        regular_invocable<internal::reinterpret_t<element_type>, U>
    __DPL_HIDE_FROM_ABI constexpr negated_mask(negated_mask<U> other) noexcept
        : negated_mask(
              dx::to_native_type(dx::reinterpret<element_type>(!other))) {}

    template <common_size_simd_with<T> U>
    requires same_abi_simd_as<T, U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit operator U(this negated_mask self) noexcept {
        return dx::bwnot(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit constexpr operator mask_type(this negated_mask self) noexcept {
        return dx::to_native_type(dx::bwnot(!self));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr mask_type operator+(this negated_mask self) noexcept
    requires basic_simd_mask_type<T>
    {
        return dx::to_native_type(dx::bwnot(!self));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr bool operator[](
        this negated_mask self, internal::extraction_index auto idx) noexcept {
        return !dx::extract(!self, idx);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr T operator!(this negated_mask self) noexcept {
        return self.mask_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr T logical_not(abi_type, negated_mask self) noexcept {
        return self.mask_;
    }

    template <dx::simd_type L, dx::simd_type R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto select(common_abi_with<abi_type> auto,
        negated_mask self, L lhs, R rhs) noexcept {
        return dx::select(!self, rhs, lhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool any_of(abi_type, negated_mask self) noexcept {
        return !dx::none_of(internal::abi<T>, !self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool all_of(abi_type, negated_mask self) noexcept {
        return dx::none_of(internal::abi<T>, !self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool none_of(abi_type, negated_mask self) noexcept {
        return dx::all_of(internal::abi<T>, !self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool some_of(abi_type, negated_mask self) noexcept {
        return dx::some_of(internal::abi<T>, !self);
    }

    template <simd_class L, common_abi_simd_with<L> R,
        same_as<common_abi_t<L, R, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto select(A, negated_mask self, L lhs, R rhs) noexcept
    requires requires { dx::select(!self, rhs, lhs); }
    {
        return dx::select(!self, rhs, lhs);
    }

    template <simd_class L, typename R, same_as<common_abi_t<L, T>> A>
    requires (!simd_class<R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto select(A, negated_mask self, L lhs, R rhs) noexcept
    requires requires { dx::select(!self, rhs, lhs); }
    {
        return dx::select(!self, rhs, lhs);
    }

    template <typename L, common_abi_simd_with<L> R,
        same_as<common_abi_t<R, T>> A>
    requires (!simd_class<L>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto select(A, negated_mask self, L lhs, R rhs) noexcept
    requires requires { dx::select(!self, rhs, lhs); }
    {
        return dx::select(!self, rhs, lhs);
    }

    template <broadcastable_to<T> L, broadcastable_to<T> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto select(
        abi_type, negated_mask self, L lhs, R rhs) noexcept
    requires requires { dx::select(!self, rhs, lhs); }
    {
        return dx::select(!self, rhs, lhs);
    }

    template <common_size_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwand(
        A, negated_mask self, negated_mask<U> arg) noexcept
    requires requires { !dx::bwor(!self, !arg); }
    {
        return !dx::bwor(!self, !arg);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwnot(abi_type, negated_mask self) noexcept {
        return !self;
    }

    template <common_size_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwor(
        A, negated_mask self, negated_mask<U> arg) noexcept
    requires requires { !dx::bwand(!self, !arg); }
    {
        return !dx::bwand(!self, !arg);
    }

    template <common_size_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwxor(
        A, negated_mask self, negated_mask<U> arg) noexcept
    requires requires { dx::bwxor(!self, !arg); }
    {
        return dx::bwxor(!self, !arg);
    }

    template <common_size_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwandnot(
        A, negated_mask self, negated_mask<U> arg) noexcept
    requires requires { dx::bwandnot(!arg, !self); }
    {
        return dx::bwandnot(!arg, !self);
    }

    template <common_size_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwornot(
        A, negated_mask self, negated_mask<U> arg) noexcept
    requires requires { dx::bwornot(!arg, !self); }
    {
        return dx::bwornot(!arg, !self);
    }

    template <simd_mask_type M, same_as<common_abi_t<M, T>> A>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwand(A, negated_mask self, M arg) noexcept
    requires requires { dx::bwandnot(arg, !self); }
    {
        return dx::bwandnot(arg, !self);
    }

    template <simd_mask_type M, same_as<common_abi_t<M, T>> A>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwor(A, negated_mask self, M arg) noexcept
    requires requires { dx::bwornot(arg, !self); }
    {
        return dx::bwornot(arg, !self);
    }

    template <simd_mask_type M, same_as<common_abi_t<M, T>> A>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwxor(A, negated_mask self, M arg) noexcept
    requires requires { !dx::bwxor(!self, arg); }
    {
        return !dx::bwxor(!self, arg);
    }

    template <simd_mask_type M, same_as<common_abi_t<M, T>> A>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwandnot(A, negated_mask self, M arg) noexcept
    requires requires { !dx::bwor(!arg, self); }
    {
        return !dx::bwor(!arg, self);
    }

    template <simd_mask_type M, same_as<common_abi_t<M, T>> A>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwornot(A, negated_mask self, M arg) noexcept
    requires requires { !dx::bwand(!arg, self); }
    {
        return !dx::bwand(!arg, self);
    }

    template <common_size_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_keep(
        A, negated_mask self, negated_mask<U> arg) noexcept
    requires requires { !dx::bit_fill(!self, !arg); }
    {
        return !dx::bit_fill(!self, !arg);
    }

    template <common_size_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_drop(
        A, negated_mask self, negated_mask<U> arg) noexcept
    requires requires { !dx::bit_stencil(!self, !arg); }
    {
        return !dx::bit_stencil(!self, !arg);
    }

    template <common_size_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_fill(
        A, negated_mask self, negated_mask<U> arg) noexcept
    requires requires { !dx::bit_keep(!self, !arg); }
    {
        return !dx::bit_keep(!self, !arg);
    }

    template <common_size_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_stencil(
        A, negated_mask self, negated_mask<U> arg) noexcept
    requires requires { !dx::bit_drop(!self, !arg); }
    {
        return !dx::bit_drop(!self, !arg);
    }

    template <common_abi_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    requires (!internal::is_negated_mask_specialization<U>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_keep(A, negated_mask self, U arg) noexcept
    requires requires { dx::bit_drop(!self, arg); }
    {
        return dx::bit_drop(!self, arg);
    }

    template <common_abi_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    requires (!internal::is_negated_mask_specialization<U>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_drop(A, negated_mask self, U arg) noexcept
    requires requires { datapar::bit_keep(!self, arg); }
    {
        return dx::bit_keep(!self, arg);
    }

    template <common_abi_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    requires (!internal::is_negated_mask_specialization<U>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_fill(A, negated_mask self, U arg) noexcept
    requires requires { dx::bit_stencil(!self, arg); }
    {
        return dx::bit_stencil(!self, arg);
    }

    template <common_abi_simd_with<T> U, same_as<common_abi_t<U, T>> A>
    requires (!internal::is_negated_mask_specialization<U>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_stencil(A, negated_mask self, U arg) noexcept
    requires requires { dx::bit_fill(!self, arg); }
    {
        return dx::bit_fill(!self, arg);
    }

    template <simd_class L, common_abi_simd_with<L> R,
        same_as<common_abi_t<L, R, T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_select(
        A, negated_mask self, L lhs, R rhs) noexcept
    requires requires { dx::bit_select(!self, rhs, lhs); }
    {
        return dx::bit_select(!self, rhs, lhs);
    }

    template <simd_class L, typename R, same_as<common_abi_t<L, T>> A>
    requires (!simd_class<R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_select(
        A, negated_mask self, L lhs, R rhs) noexcept
    requires requires { dx::bit_select(!self, rhs, lhs); }
    {
        return dx::bit_select(!self, rhs, lhs);
    }

    template <typename L, common_abi_simd_with<L> R,
        same_as<common_abi_t<R, T>> A>
    requires (!simd_class<L>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_select(
        A, negated_mask self, L lhs, R rhs) noexcept
    requires requires { dx::bit_select(!self, rhs, lhs); }
    {
        return dx::bit_select(!self, rhs, lhs);
    }

    template <broadcastable_to<T> L, broadcastable_to<T> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bit_select(
        abi_type, negated_mask self, L lhs, R rhs) noexcept
    requires requires { dx::bit_select(!self, rhs, lhs); }
    {
        return dx::bit_select(!self, rhs, lhs);
    }

private:
    mask_type mask_;
};

template <simd_mask_type T>
inline constexpr bool is_negated_mask_specialization<negated_mask<T>> = true;
} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <simd_mask_type T>
inline constexpr bool enable_simd_mask<internal::negated_mask<T>> = true;
}
DPL_DEFAULT_NAMESPACE_END
