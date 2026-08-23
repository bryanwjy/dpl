// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcast.h"
#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/from_bitset.h"
#include "dpl/core/basic/initializers.h"
#include "dpl/core/basic/to_bitset.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/utility/bitset.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename E, typename A>
class basic_mask {
    static_assert(is_same_v<E, decay_t<E>> && is_same_v<A, decay_t<A>>);
    static_assert(simd_element_for<E, A> &&
            is_same_v<simd_element_representation_t<A, E>, E>,
        "Unsupported element type");
    using abi_traits DPL_NODEBUG = simd_abi_traits<E, A>;
    using mask_type DPL_NODEBUG = typename abi_traits::native_mask;

    template <typename E2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_mask reinterpret(basic_mask<E2, A> other) noexcept {
        using other_mask = typename simd_abi_traits<E2, A>::native_mask;
        if constexpr (same_as<mask_type, other_mask>) {
            return basic_mask(+other);
        } else if consteval {
            if constexpr (fixed_width_abi<A>) {
                return datapar::from_bitset<E, A>(datapar::to_bitset(other));
            }
        }

        return __DPL bit_cast<mask_type>(+other);
    }

public:
    using vector_type = basic_vector<E, A>;
    using value_type = bool;
    using abi_type = A;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t size() noexcept {
        return abi_traits::size();
    }

    __DPL_HIDE_FROM_ABI constexpr basic_mask() noexcept
        : basic_mask(datapar::broadcast<basic_mask>(false_type{})) {}

    __DPL_HIDE_FROM_ABI explicit constexpr basic_mask(
        broadcasting_t, bool scalar) noexcept
        : basic_mask(datapar::broadcast<basic_mask>(scalar)) {}

    __DPL_HIDE_FROM_ABI constexpr basic_mask(mask_type data) noexcept
        : mask_{data} {}

    __DPL_HIDE_FROM_ABI constexpr basic_mask(
        bitset<abi_traits::size> data) noexcept
    requires fixed_width_abi<A>
        : basic_mask(datapar::from_bitset<E, A>(data)) {}

    template <size_t W, internal::mask_value_t<W> V>
    requires fixed_width_abi<A> &&
        requires { typename const_mask<abi_traits::size, V>; }
    __DPL_HIDE_FROM_ABI explicit(!convertible_to<const_mask<W, V>,
        const_mask<abi_traits::size, V>>) constexpr basic_mask(const_mask<W, V>
            mask) noexcept
        : basic_mask(
              datapar::to_bitset(const_mask<abi_traits::size, V>(mask))) {}

    template <core_convertible_to<bool>... Bs>
    requires fixed_width_abi<A> && (sizeof...(Bs) == abi_traits::size)
    __DPL_HIDE_FROM_ABI explicit constexpr basic_mask(Bs&&... args) noexcept
        : basic_mask(bitset<abi_traits::size>(
              static_cast<bool>(__DPL forward<Bs>(args))...)) {}

    template <different_from<E> E2>
    requires simd_element_for<E2, A> && common_size_with<E, E2>
    __DPL_HIDE_FROM_ABI constexpr basic_mask(basic_mask<E2, A> other) noexcept
        : basic_mask(reinterpret(other)) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr mask_type operator+(this basic_mask self) noexcept {
        return self.mask_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr bool operator[](
        this basic_mask self, internal::extraction_index auto idx) noexcept {
        // assert(idx < simd_abi_traits<basic_mask>::size);
        return datapar::extract(self, idx);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit constexpr operator mask_type(this basic_mask self) noexcept {
        return self.mask_;
    }

private:
    mask_type mask_;
};

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
