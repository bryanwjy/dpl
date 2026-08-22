// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/basic_vector.h" // IWYU pragma: keep
#  include "dpl/core/basic/gather.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/dispatch/concepts/operation.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dx = __DPL datapar;

template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E>
requires (same_as<int, simd_element_type_t<I>> ||
             same_as<ptrdiff_t, simd_element_type_t<I>>) &&
    (dx::simd_canonical_invocable<dx::internal::gather_t, E const*, I> ||
        fixed_width_abi<simd_abi_type_t<I>>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr rebind_simd_t<I, E>
    DPL_VECTORCALL gather(E const* ptr, I idx) noexcept {
    if constexpr (requires { dx::gather(ptr, idx); }) {
        return dx::gather(ptr, idx);
    } else {
        using A = simd_abi_type_t<I>;
        return [&]<size_t... Is>(index_sequence<Is...>) {
            constexpr auto size = sizeof...(Is);
            return dx::initialize<E, A>([&]<size_t J>(immediate<J> imm) {
                if constexpr (J < simd_abi_traits<I>::size()) {
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
__DPL_DEFAULT_NAMESPACE_END
