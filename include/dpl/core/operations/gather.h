// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/to_basic_type.h"
#  include "dpl/core/concepts/integral_simd.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void gather(...) noexcept = delete;

template <typename E, typename I>
concept unqualified_gather = requires(E const* ptr, I idx) {
    {
        gather(internal::abi<I>, ptr, idx)
    } -> simd_with<E, typename I::abi_type>;
};

struct gather_t {
private:
    template <typename E, typename I, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<E, A> fallback(
        E const* ptr, basic_simd<I, A> idx) noexcept {
        return []<size_t... Is>(
                   E const* ptr, basic_simd<I, A> idx, index_sequence<Is...>) {
            return dx::initialize<E>(internal::abi<A>, ptr[idx[imm<Is>]]...);
        }(ptr, idx, iota_sequence<E, A>);
    }

public:
    template <simd_element E, integral_simd I>
    requires basic_simd_type<rebind_simd_t<I, E>> &&
        (sizeof(E) >= sizeof(typename I::value_type))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    static constexpr rebind_simd_t<I, E> DPL_VECTORCALL operator()(
        E const* ptr, I idx) noexcept {
        if constexpr (unqualified_gather<E, I>) {
            if constexpr (basic_simd_type<I>) {
                if consteval {
                    return fallback(ptr, idx);
                } else {
                    return gather(internal::abi<I>, ptr, idx);
                }
            } else {
                return gather(internal::abi<I>, ptr, idx);
            }
        } else if constexpr (basic_simd_type<I>) {
            return fallback(ptr, idx);
        } else {
            return operator()(ptr, dx::to_basic_type(idx));
        }
    }

    template <simd_element E, integral_simd I>
    requires (sizeof(E) >= sizeof(typename I::value_type)) &&
        (!basic_simd_type<rebind_simd_t<I, E>>) &&
        (unqualified_gather<E, I> || unqualified_gather<E, basic_type_t<I>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    static constexpr rebind_simd_t<I, E> DPL_VECTORCALL operator()(
        E const* ptr, I idx) noexcept {
        if constexpr (unqualified_gather<E, I>) {
            return gather(internal::abi<I>, ptr, idx);
        } else {
            return gather(internal::abi<I>, ptr, dx::to_basic_type(idx));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::gather_t gather{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
