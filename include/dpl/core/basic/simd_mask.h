// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcast.h"
#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/immediate.h"
#include "dpl/core/basic/initialize.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element_for.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/type_traits/is_class.h"
#  include "dpl/std/type_traits/is_union.h"
#  include "dpl/std/type_traits/remove_all_extents.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename E, simd_abi A>
requires simd_element_for<E, A>
class basic_simd_mask<E, A> {
    using traits DPL_NODEBUG = simd_abi_traits<E, A>;
    using mask_type DPL_NODEBUG = typename traits::native_mask;
    using vector_type DPL_NODEBUG = typename traits::native_type;

    template <typename E2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_simd_mask reinterpret(
        basic_simd_mask<E2, A> other) noexcept {
        // Implement using bit_cast to avoid making reinterpret a
        // basic-operation
        using other_mask = typename simd_abi_traits<E2, A>::native_mask;
        if constexpr (same_as<mask_type, other_mask>) {
            return basic_simd_mask(+other);
        } else if consteval {
            if constexpr (fixed_width_abi<A>) {
                return [&]<size_t... Is>(index_sequence<Is...>) {
                    return datapar::initialize<E, A>(other[imm<Is>]...);
                }(iota_sequence<E, A>);
            }
        }

        return __DPL bit_cast<mask_type>(+other);
    }

public:
    using simd_type = basic_simd<E, A>;
    using value_type = bool;
    using abi_type = A;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t size() noexcept {
        if constexpr (fixed_width_abi<A>) {
            return traits::size;
        } else {
            return traits::size();
        }
    }

    __DPL_HIDE_FROM_ABI constexpr basic_simd_mask() noexcept
        : basic_simd_mask(datapar::broadcast<basic_simd_mask>(false)) {}

    __DPL_HIDE_FROM_ABI constexpr basic_simd_mask(mask_type data) noexcept
        : mask_{data} {}

    template <core_convertible_to<bool>... Bs>
    requires fixed_width_abi<A> && (sizeof...(Bs) == traits::size)
    __DPL_HIDE_FROM_ABI constexpr basic_simd_mask(Bs&&... args) noexcept
        : basic_simd_mask(datapar::initialize<A>(
              static_cast<bool>(__DPL forward<Bs>(args))...)) {}

    template <different_from<E> E2>
    requires simd_element_for<E2, A> && common_size_with<E, E2>
    __DPL_HIDE_FROM_ABI constexpr basic_simd_mask(
        basic_simd_mask<E2, A> other) noexcept
        : basic_simd_mask(reinterpret(other)) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr mask_type operator+(this basic_simd_mask self) noexcept {
        return self.mask_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr bool operator[](this basic_simd_mask self,
        internal::extraction_index auto idx) noexcept {
        // assert(idx < simd_abi_traits<basic_simd_mask>::size);
        return datapar::extract(self, idx);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit constexpr operator mask_type(this basic_simd_mask self) noexcept {
        return self.mask_;
    }

private:
    mask_type mask_;
};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
