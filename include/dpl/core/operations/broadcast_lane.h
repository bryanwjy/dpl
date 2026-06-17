// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/permute.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void broadcast_lane(...) noexcept = delete;

struct DPL_EMPTY_BASES broadcast_lane_t :
    private primitive_operation_base<broadcast_lane_t>,
    private maskable_transform_base<broadcast_lane_t> {
    using operation_base<broadcast_lane_t>::operator();
    using maskable_transform_base<broadcast_lane_t>::operator();
};

template <>
struct operation_signature<broadcast_lane_t> {
    template <simd_vector L, typename R>
    requires index_sequence_like<R>
    static consteval void operator()(L&&, R) noexcept {}
    template <simd_vector L>
    static consteval void operator()(L&&, size_t) noexcept {}
};

template <>
struct fallback_impl<broadcast_lane_t> {

private:
    template <simd_vector T, integral_constant_like I>
    static consteval auto broadcast_sequence(I) noexcept {
        using L = remove_cvref_t<T>;
        return [&]<size_t... Is>(index_sequence<Is...>) {
            return __DPL index_sequence<((Is / Is) * I::value)...>{};
        }(iota_sequence<L>);
    }

    template <typename T>
    using index_vector DPL_NODEBUG =
        basic_vector<signed_representation_t<simd_element_type_t<T>>,
            simd_abi_type_t<T>>;

public:
    template <simd_vector L, integral_constant_like R>
    requires fixed_width_abi<simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static constexpr auto
    operator()(L val, R idx) noexcept
    requires cpo_invocable<permute_t, L, decltype(broadcast_sequence<L>(idx))>
    {
        constexpr auto seq = broadcast_sequence<L>(idx);
        static_assert(R::value > 0 && R::value < seq.size());
        return dx::permute(__DPL forward<L>(val), seq);
    }

    template <simd_vector L>
    requires cpo_invocable<broadcast_t<index_vector<L>>,
                 simd_element_type_t<index_vector<L>>> &&
        cpo_invocable<permute_t, L, index_vector<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        L val, size_t idx) noexcept {
        using index_t = simd_element_type_t<index_vector<L>>;
        return dx::permute( __DPL forward<L>(val),
            dx::broadcast<index_vector<L>>(static_cast<index_t>(idx)));
    }
};

template <>
struct canonical_impl<broadcast_lane_t> {
    template <canonical_vector L, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static constexpr L operator()(
        L val, R idx) noexcept
    requires requires { broadcast_lane(internal::abi<L>, val, idx); }
    {
        return broadcast_lane(internal::abi<L>, val, idx);
    }

    template <canonical_vector L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static constexpr L operator()(
        L val, size_t idx) noexcept
    requires requires { broadcast_lane(internal::abi<L>, val, idx); }
    {
        return broadcast_lane(internal::abi<L>, val, idx);
    }
};

template <typename T, typename N, typename A = typename T::abi_type>
concept unqualified_extended_broadcast_lane = requires(N idx) {
    {
        broadcast_lane(internal::declarg<T>(), idx)
    } -> vector_with_common_abi<A>;
};

template <>
struct extended_impl<broadcast_lane_t> {
    template <simd_vector L, integral_constant_like R>
    requires unqualified_extended_broadcast_lane<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& val, R idx) {
        return broadcast_lane(__DPL forward<L>(val), idx);
    }

    template <simd_vector L>
    requires unqualified_extended_broadcast_lane<L, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L&& val, size_t idx) {
        return broadcast_lane(__DPL forward<L>(val), idx);
    }
};

template <size_t I>
struct broadcast_lanei_t {
    template <typename... Ts>
    requires cpo_invocable<broadcast_lane_t, Ts..., size_constant<I>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Ts&&... args) noexcept(
        (... && (!simd_type<Ts> || canonical_simd_type<Ts>))) {
        constexpr size_constant<I> idx;
        return broadcast_lane_t::operator()(__DPL forward<Ts>(args)..., idx);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <size_t I>
inline constexpr internal::broadcast_lanei_t<I> broadcast_lanei{};
DPL_EXPORT inline constexpr internal::broadcast_lane_t broadcast_lane{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
