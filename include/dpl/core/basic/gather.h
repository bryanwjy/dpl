// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/initialize.h"
#include "dpl/core/basic/internal/abi.h"
#include "dpl/core/basic/internal/iota_sequence.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/immediate/immediate.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void gather(...) noexcept = delete;

struct gather_t : private basic_operation_base<gather_t> {
    using operation_base<gather_t>::operator();
    // TODO maskable
};

template <>
struct operation_signature<gather_t> {
    template <canonical_vector T>
    requires integral<simd_element_type_t<T>>
    static consteval void operator()(
        simd_element_for<simd_abi_type_t<T>> auto const*, T) noexcept {}
};

template <>
struct fallback_impl<gather_t> {
    template <fixed_width_abi A, simd_element_for<A> I, simd_element_for<A> E>
    requires integral<I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        E const* ptr, basic_vector<I, A> idx) noexcept {
        return []<size_t... Is>(E const* ptr, basic_vector<I, A> idx,
                   index_sequence<Is...>) {
            constexpr auto size = sizeof...(Is);
            auto const zero = E();
            return dx::initialize<E, A>(
                (Is < idx.size() ? ptr[idx[imm<Is>]] : zero)...);
        }(ptr, idx, iota_sequence<E, A>);
    }
};

template <>
struct canonical_impl<gather_t> {
    template <fixed_width_abi A, simd_element_for<A> I, simd_element_for<A> E>
    requires integral<I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        E const* ptr, basic_vector<I, A> idx) noexcept
    requires requires { gather(internal::abi<I>, ptr, idx); }
    {
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
