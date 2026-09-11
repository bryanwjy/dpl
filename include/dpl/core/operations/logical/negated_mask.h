// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/basic/extract.h"
#  include "dpl/core/basic/to_native_type.h"
#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/core/type_traits/simd_native_type.h"
#endif

#include "dpl/core/operations/bit.h"
#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/logical/all_of.h"
#include "dpl/core/operations/logical/any_of.h"
#include "dpl/core/operations/logical/none_of.h"
#include "dpl/core/operations/logical/some_of.h"
#include "dpl/core/operations/reinterpret.h"
#include "dpl/core/operations/select.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename T>
inline constexpr bool is_negated_mask_specialization = false;

/**
 * Only allowed for canonical masks to to keep it simple
 */
template <canonical_mask T>
class negated_mask;

template <canonical_mask T>
inline constexpr bool is_negated_mask_specialization<negated_mask<T>> = true;

template <canonical_mask T>
requires different_from<T, simd_native_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr negated_mask<T> make_negated_mask(T val) noexcept {
    auto mmask = dx::to_native_type(val);
    return negated_mask<T>(mmask);
}

template <canonical_mask T>
requires different_from<T, simd_native_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto make_negated_mask(negated_mask<T> const& val) noexcept {
    return !val;
}

template <canonical_mask T>
class negated_mask : public simd_mask_base<negated_mask<T>> {
    using element_type DPL_NODEBUG = simd_element_type_t<T>;

public:
    using vector_type = simd_vector_type_t<T>;
    using abi_type = simd_abi_type_t<T>;
    using value_type = bool;
    using result_type = T;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr size_t size() noexcept {
        return simd_abi_traits<T>::size();
    }

    __DPL_HIDE_FROM_ABI constexpr negated_mask() noexcept : mask_() {}

    __DPL_HIDE_FROM_ABI constexpr negated_mask(
        simd_native_type_t<T> native) noexcept
        : mask_(native) {}

    template <different_from<T> U>
    requires common_mask_with<T, U> &&
        cpo_invocable<reinterpret_t<element_type>, U>
    __DPL_HIDE_FROM_ABI constexpr negated_mask(negated_mask<U> other) noexcept
        : negated_mask(dx::reinterpret<element_type>(!other)) {}

    template <different_from<T> U>
    requires convertible_to<T, U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit operator U(this negated_mask self) noexcept {
        return static_cast<U>(dx::bwnot(!self));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr T evaluate(this negated_mask self) noexcept {
        return dx::bwnot(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit constexpr operator simd_native_type_t<T>(
        this negated_mask self) noexcept {
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
    friend constexpr T logical_not(negated_mask self) noexcept {
        return self.mask_;
    }

    template <typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto select(negated_mask self, L lhs, R rhs) noexcept
    requires requires { dx::select(!self, rhs, lhs); }
    {
        return dx::select(!self, rhs, lhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool any_of(negated_mask self) noexcept {
        return internal::make_negated_mask(dx::none_of(!self));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool all_of(negated_mask self) noexcept {
        return dx::none_of(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool none_of(negated_mask self) noexcept {
        return dx::all_of(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr bool some_of(negated_mask self) noexcept {
        return dx::some_of(!self);
    }

    template <simd_mask U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwand(negated_mask self, negated_mask<U> arg) noexcept
    requires requires { internal::make_negated_mask(dx::bwor(!self, !arg)); }
    {
        return internal::make_negated_mask(dx::bwor(!self, !arg));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwnot(negated_mask self) noexcept {
        return !self;
    }

    template <simd_mask U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwor(negated_mask self, negated_mask<U> arg) noexcept
    requires requires { internal::make_negated_mask(dx::bwand(!self, !arg)); }
    {
        return internal::make_negated_mask(dx::bwand(!self, !arg));
    }

    template <simd_mask U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwxor(negated_mask self, negated_mask<U> arg) noexcept
    requires requires { dx::bwxor(!self, !arg); }
    {
        return dx::bwxor(!self, !arg);
    }

    template <simd_mask U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwandnot(
        negated_mask self, negated_mask<U> arg) noexcept
    requires requires { dx::bwandnot(!arg, !self); }
    {
        return dx::bwandnot(!arg, !self);
    }

    template <simd_mask U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwornot(
        negated_mask self, negated_mask<U> arg) noexcept
    requires requires { dx::bwornot(!arg, !self); }
    {
        return dx::bwornot(!arg, !self);
    }

    template <simd_mask M>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwand(negated_mask self, M arg) noexcept
    requires requires { dx::bwandnot(arg, !self); }
    {
        return dx::bwandnot(arg, !self);
    }

    template <simd_mask M>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwor(negated_mask self, M arg) noexcept
    requires requires { dx::bwornot(arg, !self); }
    {
        return dx::bwornot(arg, !self);
    }

    template <simd_mask M>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwxor(negated_mask self, M arg) noexcept
    requires requires { internal::make_negated_mask(dx::bwxor(!self, arg)); }
    {
        return internal::make_negated_mask(dx::bwxor(!self, arg));
    }

    template <simd_mask M>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwandnot(negated_mask self, M arg) noexcept
    requires requires {
        internal::make_negated_mask(dx::bwor(negated_mask<M>(arg), self));
    }
    {
        return internal::make_negated_mask(
            dx::bwor(negated_mask<M>(arg), self));
    }

    template <simd_mask M>
    requires (!internal::is_negated_mask_specialization<M>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto bwornot(negated_mask self, M arg) noexcept
    requires requires {
        internal::make_negated_mask(dx::bwand(negated_mask<M>(arg), self));
    }
    {
        return internal::make_negated_mask(
            dx::bwand(negated_mask<M>(arg), self));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto countl_zero(negated_mask self) noexcept
    requires requires { dx::countl_one(!self); }
    {
        return dx::countl_one(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto countl_one(negated_mask self) noexcept
    requires requires { dx::countl_zero(!self); }
    {
        return dx::countl_zero(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto countr_zero(negated_mask self) noexcept
    requires requires { dx::countr_one(!self); }
    {
        return dx::countr_one(!self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto countr_one(negated_mask self) noexcept
    requires requires { dx::countr_zero(!self); }
    {
        return dx::countr_zero(!self);
    }

    template <simd_element_for<abi_type> TE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr auto reinterpret(negated_mask self) noexcept
    requires requires { dx::reinterpret<TE>(!self); }
    {
        return internal::make_negated_mask(dx::reinterpret<TE>(!self));
    }

private:
    simd_native_type_t<T> mask_;
};

template <canonical_mask T>
requires (!is_negated_mask_specialization<T>)
negated_mask(T) -> negated_mask<T>;

} // namespace datapar::internal

namespace datapar {
template <simd_mask T, typename E, typename A>
requires requires { typename rebind_simd<T, E, A>::type; }
struct rebind_simd<internal::negated_mask<T>, E, A> {
    using type DPL_NODEBUG =
        internal::negated_mask<typename rebind_simd<T, E, A>::type>;
};
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
