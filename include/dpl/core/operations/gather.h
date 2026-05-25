// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/to_canonical.h"
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
    static constexpr basic_vector<E, A> fallback(
        E const* ptr, basic_vector<I, A> idx) noexcept {
        return []<size_t... Is>(E const* ptr, basic_vector<I, A> idx,
                   index_sequence<Is...>) {
            return dx::initialize<E>(internal::abi<A>, ptr[idx[imm<Is>]]...);
        }(ptr, idx, iota_sequence<E, A>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> I>
    requires integral<I> && (sizeof(E) >= sizeof(I))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(
            E const* ptr, basic_vector<I, A> idx) noexcept {
        if constexpr (unqualified_gather<E, I>) {
            if consteval {
                return fallback(ptr, idx);
            } else {
                return gather(internal::abi<I>, ptr, idx);
            }
        } else {
            return fallback(ptr, idx);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, simd_element_for<A> I>
    requires integral<I> && (sizeof(E) >= sizeof(I)) && unqualified_gather<E, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(
            E const* ptr, basic_vector<I, A> idx) noexcept {
        return gather(internal::abi<I>, ptr, idx);
    }

    template <typename E, simd_vector I>
    requires simd_element_for<E, typename I::abi_type> &&
        integral<typename I::value_type> &&
        (sizeof(E) >= sizeof(typename I::value_type)) &&
        unqualified_gather<E, I> ||
        decayable_vector_for<L, operation_category::lane_agnostic> &&
            regular_invocable<gather_t, E const*, canonical_type_t<I>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
        static constexpr auto DPL_VECTORCALL operator()(
            E const* ptr, I idx) noexcept {
        if constexpr (unqualified_gather<E, I>) {
            return gather(internal::abi<I>, ptr, idx);
        } else {
            return operator()(ptr, dx::to_canonical(idx));
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
