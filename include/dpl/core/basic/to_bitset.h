// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/utility/bitset.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void to_bitset(...) noexcept = delete;

struct to_bitset_t : public basic_operation_base<to_bitset_t> {
    using operation_base<to_bitset_t>::operator();

    template <const_mask_like M>
    requires (!simd_mask<M>)
    static consteval auto operator()(M mask) noexcept
    requires requires {
        { to_bitset(mask) } -> bitset_type;
    }
    {
        return to_bitset(mask);
    }
};

template <>
struct operation_signature<to_bitset_t> {
    template <simd_mask T>
    static consteval void operator()(T&&) noexcept {}
};

template <typename T>
concept unqualified_canonical_to_bitset = requires {
    { to_bitset(internal::abi<T>, internal::declarg<T>()) } -> bitset_type;
};

template <>
struct canonical_impl<to_bitset_t> {

    template <canonical_mask M>
    requires fixed_width_abi<simd_abi_type_t<M>> &&
        unqualified_canonical_to_bitset<M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bitset<simd_abi_traits<M>::size> operator()(
        M val) noexcept {
        return to_bitset(internal::abi<M>, val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::to_bitset_t to_bitset{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
