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
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/utility/ignore.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void load(...) noexcept = delete;

template <typename T, typename U = void>
struct load_t :
    public basic_operation_base<load_t<T, U>>,
    public maskable_transform_base<load_t<T, U>> {
    using operation_base<load_t<T, U>>::operator();
    using maskable_transform_base<load_t<T, U>>::operator();
};

template <typename T, typename U>
struct operation_signature<load_t<T, U>> {
    static consteval void operator()(void const*) noexcept
    requires ((same_as<void, U> && (simd_type<T> || simd_abi<T>)) ||
        (simd_abi<T> && simd_element_for<U, T>) ||
        (simd_abi<U> && simd_element_for<T, U>))
    {}

    static consteval void operator()(aligned_t, void const*) noexcept
    requires ((same_as<void, U> && (simd_type<T> || simd_abi<T>)) ||
        (simd_abi<T> && simd_element_for<U, T>) ||
        (simd_abi<U> && simd_element_for<T, U>))
    {}
};

template <typename A, typename... Ts>
concept unqualified_load =
    requires { load(internal::abi<A>, internal::declarg<Ts>()...); };

template <typename T, different_from<void> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct canonical_impl<load_t<T, U>> {
private:
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;

    using vector_t DPL_NODEBUG = make_canonical_vector_t<E, A>;
    using mask_t DPL_NODEBUG = make_canonical_mask_t<E, A>;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(E const* data) noexcept
    requires unqualified_load<A, E const*>
    {
        return load(internal::abi<A>, data);
    }

    template <const_mask_for<vector_t> M>
    requires unqualified_load<A, vector_t, launder_cmask_t<vector_t, M>,
        E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(
        vector_t src, M cmask, E const* data) noexcept {
        return load(
            internal::abi<A>, src, dx::to_const_mask<vector_t>(cmask), data);
    }

    template <const_mask_for<vector_t> M>
    requires unqualified_load<A, dx::zero_t, launder_cmask_t<vector_t, M>,
        E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(
        dx::zero_t zero, M cmask, E const* data) noexcept {
        return load(
            internal::abi<A>, zero, dx::to_const_mask<vector_t>(cmask), data);
    }

    template <const_mask_for<vector_t> M>
    requires unqualified_load<A, vector_t, launder_cmask_t<vector_t, M>,
        dx::aligned_t, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(
        vector_t src, M cmask, aligned_t aligned, E const* data) noexcept {
        return load(internal::abi<A>, src, dx::to_const_mask<vector_t>(cmask),
            aligned, data);
    }

    template <const_mask_for<vector_t> M>
    requires unqualified_load<A, dx::zero_t, launder_cmask_t<vector_t, M>,
        aligned_t, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(
        dx::zero_t zero, M cmask, aligned_t aligned, E const* data) noexcept {
        return load(internal::abi<A>, zero, dx::to_const_mask<vector_t>(cmask),
            aligned, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(
        aligned_t aligned, E const* data) noexcept
    requires unqualified_load<A, aligned_t, E const*>
    {
        return load(internal::abi<A>, aligned, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(
        vector_t src, mask_t mask, E const* data) noexcept
    requires unqualified_load<vector_t, mask_t, E const*>
    {
        return load(internal::abi<A>, src, mask, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(
        dx::zero_t zero, mask_t mask, E const* data) noexcept
    requires unqualified_load<dx::zero_t, mask_t, E const*>
    {
        return load(internal::abi<A>, zero, mask, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(
        vector_t src, mask_t mask, aligned_t aligned, E const* data) noexcept
    requires unqualified_load<vector_t, mask_t, aligned_t, E const*>
    {
        return load(internal::abi<A>, src, mask, aligned, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t operator()(
        dx::zero_t zero, mask_t mask, aligned_t aligned, E const* data) noexcept
    requires unqualified_load<dx::zero_t, mask_t, aligned_t, E const*>
    {
        return load(internal::abi<A>, zero, mask, aligned, data);
    }
};

template <simd_abi A>
struct canonical_impl<load_t<A>> {
private:
    template <typename E>
    using base_type DPL_NODEBUG = load_t<A, E>;

public:
    template <simd_element_for<A> E>
    requires cpo_invocable<base_type<E>, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(E const* data) noexcept {
        return base_type<E>::operator()(data);
    }

    template <simd_element_for<A> E>
    requires cpo_invocable<base_type<E>, aligned_t, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(aligned_t tag, E const* data) noexcept {
        return base_type<E>::operator()(tag, data);
    }

    template <typename S, typename M, simd_element_for<A> E>
    requires cpo_invocable<base_type<E>, S, M, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, E const* data) noexcept {
        return base_type<E>::operator()(
            __DPL forward<S>(src), __DPL forward<M>(mask), data);
    }

    template <typename S, typename M, simd_element_for<A> E>
    requires cpo_invocable<base_type<E>, S, M, dx::aligned_t, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, dx::aligned_t aligned, E const* data) noexcept {
        return base_type<E>::operator()(
            __DPL forward<S>(src), __DPL forward<M>(mask), aligned, data);
    }
};

template <simd_vector T>
struct canonical_impl<load_t<T>> :
    canonical_impl<load_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {};

template <typename T, typename U = void>
struct aligned_load_t {
    template <typename E>
    requires cpo_invocable<load_t<T, U>, aligned_t, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(E const* ptr) noexcept {
        return load_t<T, U>::operator()(dx::aligned, ptr);
    }

    template <typename E, typename S, typename M>
    requires cpo_invocable<load_t<T, U>, S, M, aligned_t, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, E const* ptr) noexcept {
        return load_t<T, U>::operator()(src, mask, dx::aligned, ptr);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <typename T, typename U = void>
inline constexpr internal::load_t<T, U> load{};
template <typename T, typename U = void>
inline constexpr internal::aligned_load_t<T, U> aligned_load{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
