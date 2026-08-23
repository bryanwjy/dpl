// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/aligned.h"
#include "dpl/core/basic/internal/abi.h"
#include "dpl/core/basic/internal/iota_sequence.h"
#include "dpl/core/basic/load.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/type_traits/canonical_type.h"
#  include "dpl/core/type_traits/representation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
/**
 * Lane indices must ALWAYS return canonical simd types
 */
template <typename>
void undefined(...) noexcept = delete;

template <typename T, typename U = void>
struct undefined_t : public basic_operation_base<undefined_t<T, U>> {
    using operation_base<undefined_t<T, U>>::operator();
};

template <typename T, typename U>
struct operation_signature<undefined_t<T, U>> {
    static consteval void operator()() noexcept
    requires ((same_as<void, U> && (simd_vector<T> || simd_abi<T>)) ||
        (simd_abi<T> && simd_element_for<U, T>) ||
        (simd_abi<U> && simd_element_for<T, U>))
    {}
};

template <typename E, typename A>
concept unqualified_canonical_undefined = requires {
    {
        undefined<E>(internal::abi<A>)
    } -> same_as<make_canonical_vector_t<E, A>>;
};

template <typename T, different_from<void> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct canonical_impl<undefined_t<T, U>> {
private:
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr make_canonical_vector_t<E, A> operator()() noexcept
    requires requires { undefined<E>(internal::abi<A>); }
    {
        return undefined<E>(internal::abi<A>);
    }
};

template <simd_abi A>
struct canonical_impl<undefined_t<A>> :
    canonical_impl<undefined_t<A, signed char>> {};

template <simd_vector T>
struct canonical_impl<undefined_t<T>> :
    canonical_impl<undefined_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <typename T, typename U = void>
inline constexpr internal::undefined_t<T, U> undefined{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
