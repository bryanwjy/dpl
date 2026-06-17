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

template <typename T, typename U = __DPL ignore_t>
struct load_t :
    private basic_operation_base<load_t<T, U>>,
    private maskable_transform_base<load_t<T, U>> {
    using operation_base<load_t<T, U>>::operator();
    using maskable_transform_base<load_t<T, U>>::operator();
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

template <typename A, typename E>
struct fixed_width_canonical {
private:
    class private_t {
        constexpr private_t() noexcept = default;
    };

protected:
    consteval void operator()(private_t) noexcept {}
};

template <typename A, typename... Ts>
concept unqualified_load =
    requires { load(internal::abi<A>, internal::declarg<Ts>()...); };

template <fixed_width_abi A, simd_element_for<A> E>
struct fixed_width_canonical<A, E> {
private:
    static constexpr auto size = simd_abi_traits<A, E>::size();
    using imask_t DPL_NODEBUG = mask_value_t<size>;

    template <imask_t M>
    using cmask_t DPL_NODEBUG = const_mask<size, M>;

protected:
    template <imask_t M>
    requires unqualified_load<A, basic_vector<E, A>, cmask_t<M>, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> src, cmask_t<M> cmask, E const* data) noexcept {
        return load(internal::abi<A>, src, cmask, data);
    }

    template <imask_t M>
    requires unqualified_load<A, dx::zero_t, cmask_t<M>, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        dx::zero_t zero, cmask_t<M> cmask, E const* data) noexcept {
        return load(internal::abi<A>, zero, cmask, data);
    }

    template <imask_t M>
    requires unqualified_load<A, basic_vector<E, A>, cmask_t<M>, dx::aligned_t,
        E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        cmask_t<M> cmask, aligned_t aligned, E const* data) noexcept {
        return load(internal::abi<A>, src, cmask, aligned, data);
    }

    template <imask_t M>
    requires unqualified_load<A, dx::zero_t, cmask_t<M>, aligned_t, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(dx::zero_t zero,
        cmask_t<M> cmask, aligned_t aligned, E const* data) noexcept {
        return load(internal::abi<A>, zero, cmask, aligned, data);
    }
};

template <simd_abi A, simd_element_for<A> E>
struct canonical_impl<load_t<A, E>> : private fixed_width_canonical<A, E> {
    using fixed_width_canonical<A, E>::operator();

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(E const* data) noexcept
    requires unqualified_load<A, E const*>
    {
        return load(internal::abi<A>, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        aligned_t aligned, E const* data) noexcept
    requires unqualified_load<A, aligned_t, E const*>
    {
        return load(internal::abi<A>, aligned, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> src, basic_mask<E, A> mask, E const* data) noexcept
    requires unqualified_load<basic_vector<E, A>, basic_mask<E, A>, E const*>
    {
        return load(internal::abi<A>, src, mask, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        dx::zero_t zero, basic_mask<E, A> mask, E const* data) noexcept
    requires unqualified_load<dx::zero_t, basic_mask<E, A>, E const*>
    {
        return load(internal::abi<A>, zero, mask, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<E, A> mask, aligned_t aligned, E const* data) noexcept
    requires unqualified_load<basic_vector<E, A>, basic_mask<E, A>, aligned_t,
        E const*>
    {
        return load(internal::abi<A>, src, mask, aligned, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(dx::zero_t zero,
        basic_mask<E, A> mask, aligned_t aligned, E const* data) noexcept
    requires unqualified_load<dx::zero_t, basic_mask<E, A>, aligned_t, E const*>
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

template <simd_type T>
struct canonical_impl<load_t<T>> :
    canonical_impl<load_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {};

template <typename T, typename U = ignore_t>
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
DPL_EXPORT template <typename T, typename U = __DPL ignore_t>
inline constexpr internal::load_t<T, U> load{};
DPL_EXPORT template <typename T, typename U = __DPL ignore_t>
inline constexpr internal::aligned_load_t<T, U> aligned_load{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
