// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void pack_mask(...) noexcept = delete;

struct pack_mask_t : private primitive_operation_base<pack_mask_t> {
    using operation_base<pack_mask_t>::operator();
};

template <>
struct operation_signature<pack_mask_t> {
    template <simd_mask T>
    static consteval void operator()(T&&) noexcept {}
};

template <>
struct fallback_impl<pack_mask_t> {
    template <fixed_width_abi A, simd_element_for<A> E>
    static constexpr auto width = simd_abi_traits<E, A>::size();
    template <fixed_width_abi A, simd_element_for<A> E>
    using bitset_t DPL_NODEBUG = bitset<width<A, E>>;

    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bitset_t<A, E>
        DPL_VECTORCALL operator()(basic_mask<E, A> val) noexcept {
        return [&]<size_t... Is>(index_sequence<Is...>) {
            return bitset_t<A, E>(val[imm<Is>]...);
        }(iota_sequence<E, A>);
    }
};

template <typename T>
concept unqualified_canonical_pack_mask = requires {
    { pack_mask(internal::abi<T>, internal::declarg<T>()) } -> bitset_type;
};

template <>
struct canonical_impl<pack_mask_t> {

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_pack_mask<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<E, A> val) noexcept {
        return pack_mask(internal::abi<A>, val);
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    requires unqualified_canonical_pack_mask<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bitset<simd_abi_traits<A, E>::size> operator()(
        basic_mask<E, A> val) noexcept {
        return pack_mask(internal::abi<A>, val);
    }
};

template <typename T>
concept unqualified_extended_pack_mask = requires {
    { pack_mask(internal::declarg<T>()) } -> bitset_type;
};

template <>
struct extended_impl<pack_mask_t> {
    template <typename T>
    requires extended_mask<remove_cvref_t<T>> &&
        unqualified_extended_pack_mask<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto extended(T&& val) {
        return pack_mask(__DPL forward<T>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::pack_mask_t pack_mask{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
