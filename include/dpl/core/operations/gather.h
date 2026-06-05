// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void gather(...) noexcept = delete;

template <typename E, typename I>
concept unqualified_gather = requires(E const* ptr, I idx) {
    {
        gather(internal::abi<I>, ptr, idx)
    } -> simd_type_with<E, simd_abi_type_t<I>>;
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
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::gather_t gather{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
