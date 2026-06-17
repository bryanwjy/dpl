// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/aligned.h"
#include "dpl/core/basic/internal/abi.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/utility/ignore.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void load(...) noexcept = delete;
void aligned_load(...) noexcept = delete;

template <typename T, typename U = __DPL ignore_t>
struct load_t : private basic_operation_base<load_t<T, U>> {
    using operation_base<load_t<T, U>>::operator();
    // TODO
    // using maskable_transform_base<load_t<T, U>>::operator();
};

template <typename T, typename U>
struct operation_signature<load_t<T, U>> {
    static consteval void operator()(void const*) noexcept
    requires ((same_as<ignore_t, U> && (simd_type<T> || simd_abi<T>)) ||
        (simd_abi<T> && simd_element_for<U, T>) ||
        (simd_abi<U> && simd_element_for<T, U>))
    {}

    static consteval void operator()(aligned_t, void const*) noexcept
    requires ((same_as<ignore_t, U> && (simd_type<T> || simd_abi<T>)) ||
        (simd_abi<T> && simd_element_for<U, T>) ||
        (simd_abi<U> && simd_element_for<T, U>))
    {}
};

template <simd_abi A, simd_element_for<A> E>
struct fallback_impl<load_t<A, E>> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        aligned_t aligned, E const* data) noexcept
    requires requires { load(internal::abi<A>, data); }
    {
        return load(internal::abi<A>, data);
    }
};

template <simd_abi A, simd_element_for<A> E>
struct canonical_impl<load_t<A, E>> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(E const* data) noexcept
    requires requires { load(internal::abi<A>, data); }
    {
        return load(internal::abi<A>, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        aligned_t aligned, E const* data) noexcept
    requires requires { load(internal::abi<A>, aligned, data); }
    {
        return load(internal::abi<A>, aligned, data);
    }
};

template <simd_abi A>
struct canonical_impl<load_t<A>> {
private:
    template <typename E>
    using base_type DPL_NODEBUG = load_t<A, E>;

public:
    template <simd_element_for<A> E>
    requires regular_invocable<base_type<E>, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(E const* src) noexcept {
        return base_type<E>::operator()(src);
    }

    template <simd_element_for<A> E>
    requires regular_invocable<base_type<E>, aligned_t, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(aligned_t tag, E const* src) noexcept {
        return base_type<E>::operator()(tag, src);
    }
};

template <simd_type T>
struct canonical_impl<load_t<T>> :
    canonical_impl<load_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {};

template <typename T, typename U = ignore_t>
struct aligned_load_t {
    template <typename E>
    requires cpo_invocable<load_t<T, U>, aligned_t, E const*> ||
        cpo_invocable<load_t<T, U>, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(E const* ptr) noexcept {
        if constexpr (cpo_invocable<load_t<T, U>, aligned_t, E const*>) {
            return load_t<T, U>::operator()(dx::aligned, ptr);
        } else {
            return load_t<T, U>::operator()(ptr);
        }
    }

    template <canonical_vector S, typename M>
    requires (canonical_mask<M> || const_mask_for<M, S>) &&
        (cpo_invocable<load_t<T, U>, S, M, aligned_t,
             simd_element_type_t<S> const*> ||
            cpo_invocable<load_t<T, U>, S, M, simd_element_type_t<S> const*>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(
        S src, M mask, simd_element_type_t<S> const* ptr) noexcept {
        if constexpr (cpo_invocable<load_t<T, U>, aligned_t,
                          simd_element_type_t<S> const*>) {
            return load_t<T, U>::operator()(src, mask, dx::aligned, ptr);
        } else {
            return load_t<T, U>::operator()(src, mask, ptr);
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename... Args>
inline constexpr internal::load_t<Args...> load{};
DPL_EXPORT template <typename... Args>
inline constexpr internal::aligned_load_t<Args...> aligned_load{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
