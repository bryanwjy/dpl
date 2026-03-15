// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cload(...) noexcept = delete;

template <typename...>
struct cload_t;

template <simd_type T>
struct cload_t<T> {
    using value_type = typename T::value_type;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(value_type const* src,
        value_type const* alt DPL_ATTRIBUTE(NONNULL)) noexcept {
        return dx::load<T>(src ? src : alt);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(value_type const* src,
        broadcastable_to<T> auto val = dx::zero) noexcept {
        return operator()(src, dx::broadcast<T>(val));
    }

    template <equivalent_simd_as<T> U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        value_type const* src, U val) noexcept {
        if consteval {
            if (src) {
                return dx::load<T>(src);
            } else {
                return dx::reinterpret<T>(val);
            }
        } else {
            using mask_type = basic_simd_mask<value_type, typename T::abi_type>;
            auto const mask = dx::broadcast<mask_type>(src != nullptr);
            auto const loaded = operator()(src);
            return dx::select(mask, loaded, val);
        }
    }
};

template <simd_abi A>
struct cload_t<A> {
    using abi_type = A;

    template <simd_element E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(
        E const* src, E const* alt DPL_ATTRIBUTE(NONNULL)) noexcept {
        return cload_t<basic_simd<E, A>>::operator()(src, alt);
    }

    template <simd_element E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(
        E const* src, broadcastable_to<A> auto val = datapar::zero) noexcept {
        return cload_t<basic_simd<E, A>>::operator()(src, val);
    }

    template <simd_element E, equivalent_simd_as<basic_simd<E, A>> U>
    requires regular_invocable<cload_t<basic_simd<E, A>>, U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(E const* src, U val) noexcept {
        return cload_t<basic_simd<E, A>>::operator()(src, val);
    }
};

template <simd_abi A, simd_element E>
struct cload_t<A, E> : cload_t<basic_simd<E, A>> {};

template <simd_element E, simd_abi A>
struct cload_t<E, A> : cload_t<basic_simd<E, A>> {};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename T>
inline constexpr internal::cload_t<T> cload{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
