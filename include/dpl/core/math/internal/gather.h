// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/basic_vector.h" // IWYU pragma: keep
#  include "dpl/core/basic/gather.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/dispatch/concepts/operation.h"
#  include "dpl/core/immediate/immediate.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dx = __DPL datapar;

template <simd_abi A, simd_element_for<A> E, simd_element_for<A> I>
requires integral<I> &&
    (dx::simd_canonical_invocable<dx::gather, E const*, basic_vector<I, A>> ||
        fixed_width_abi<A>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr basic_vector<E, A>
    DPL_VECTORCALL gather(E const* ptr, basic_vector<I, A> idx) noexcept {
    if constexpr (requires { dx::gather(ptr, idx); }) {
        return dx::gather(ptr, idx);
    } else {
        return [&]<size_t... Is>(index_sequence<Is...>) {
            constexpr auto size = sizeof...(Is);
            return dx::initialize<E>([&]<size_t J>(immediate<J> imm) {
                if constexpr (J < basic_vector<I, A>::size()) {
                    return ptr[idx[imm]];
                } else {
                    constexpr E zero{};
                    return zero;
                }
            }(imm<Is>)...);
        }(iota_sequence<E>);
    }
}

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END
