// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/aligned.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void load(...) noexcept = delete;
void aligned_load(...) noexcept = delete;

template <typename...>
struct load_t {};
template <typename...>
struct aligned_load_t {};

template <simd_abi A, simd_element_for<A> E>
struct load_t<A, E> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<E, A> operator()(E const* data) noexcept
    requires requires(E const* data) {
        { load(internal::abi<A>, data) } -> same_as<basic_vector<E, A>>;
    }
    {
        return load(internal::abi<A>, data);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        aligned_t, E const* data) noexcept {
        if consteval {
            return operator()(data);
        } else {
            if constexpr (requires { aligned_load(internal::abi<A>, data); }) {
                return aligned_load(internal::abi<A>, data);
            } else {
                return operator()(data);
            }
        }
    }
};

template <simd_abi A, simd_element_for<A> E>
struct load_t<E, A> : load_t<A, E> {};

template <canonical_vector T>
struct load_t<T> : load_t<typename T::value_type, typename T::abi_type> {};

template <simd_vector T>
struct load_t<T> {
    using E DPL_NODEBUG = typename T::value_type;
    using base_type DPL_NODEBUG = load_t<canonical_type_t<T>>;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(E const* src) noexcept
    requires regular_invocable<base_type, E const*> &&
        explicitly_convertible_to<canonical_type_t<T>, T>
    {
        return static_cast<T>(load_t<canonical_type_t<T>>::operator()(src));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(aligned_t tag, E const* src) noexcept
    requires regular_invocable<base_type, aligned_t, E const*> &&
        explicitly_convertible_to<canonical_type_t<T>, T>
    {
        return static_cast<T>(
            load_t<canonical_type_t<T>>::operator()(tag, src));
    }
};

template <simd_abi A>
struct load_t<A> {
private:
    template <typename E>
    using base_type DPL_NODEBUG = load_t<basic_vector<E, A>>;

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

template <simd_abi A, simd_element_for<A> E>
struct aligned_load_t<A, E> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(E const* src) noexcept
    requires regular_invocable<load_t<E, A>, E const*>
    {
        return load_t<E, A>::operator()(aligned, src);
    }
};

template <simd_abi A, simd_element_for<A> E>
struct aligned_load_t<E, A> : aligned_load_t<A, E> {};

template <simd_vector T>
struct aligned_load_t<T> : private load_t<T> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(typename T::value_type const* src) noexcept
    requires regular_invocable<load_t<T>, aligned_t,
        typename T::value_type const*>
    {
        return load_t<T>::operator()(aligned, src);
    }
};

template <simd_abi A>
struct aligned_load_t<A> : private load_t<A> {
    template <simd_element_for<A> E>
    requires regular_invocable<load_t<A>, aligned_t, E const*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(E const* src) noexcept {
        return load_t<A>::operator()(aligned, src);
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
