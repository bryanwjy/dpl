// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_element_for.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#  include "dpl/std/type_traits/remove_cv.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
struct simd_vector_base {
protected:
    constexpr ~simd_vector_base() = default;
};

DPL_EXPORT template <typename T>
inline constexpr bool enable_vector_type = derived_from<T, simd_vector_base<T>>;
DPL_EXPORT template <simd_abi A, simd_element_for<A> E>
inline constexpr bool enable_vector_type<basic_vector<E, A>> = true;

namespace atom {
template <typename T>
concept simd_basics = is_object_v<T> && is_trivially_copyable_v<T> && requires {
    typename T::value_type;
    typename T::abi_type;
    requires simd_abi<typename T::abi_type>;
};

template <typename T>
concept simd_vector = enable_vector_type<T> &&
    requires {
        requires simd_element_for<typename T::value_type, typename T::abi_type>;
        requires sizeof(typename T::value_type) <= T::abi_type::size;
    } &&
    explicitly_convertible_to<T,
        typename T::abi_type::template native_vector<typename T::value_type>>;
} // namespace atom

DPL_EXPORT template <typename T>
struct simd_mask_base {
protected:
    constexpr ~simd_mask_base() = default;
};

DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_mask = derived_from<T, simd_mask_base<T>>;
DPL_EXPORT template <simd_abi A, simd_element_for<A> E>
inline constexpr bool enable_simd_mask<basic_mask<E, A>> = true;

namespace atom {
template <typename M>
concept simd_mask = enable_simd_mask<M> && requires(M const mask) {
    typename M::vector_type;
    requires same_as<typename M::value_type, bool> &&
        simd_vector<typename M::vector_type> &&
        simd_basics<typename M::vector_type>;
    requires explicitly_convertible_to<M,
        typename M::abi_type::template native_mask<
            typename M::vector_type::value_type>>;
};
} // namespace atom
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
