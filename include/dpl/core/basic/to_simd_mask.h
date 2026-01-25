// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/initialize.h"
#include "dpl/core/basic/to_basic_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/bit_type.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/type_traits/to_simd_mask_type.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/type_traits/is_enum.h"
#  include "dpl/std/utility/sequence.h"
#  include "dpl/std/utility/to_signed.h"
#  include "dpl/std/utility/to_underlying.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
/**
 * Tag type to request direct 'fast' conversions if supported
 */
DPL_EXPORT
struct direct_t {
    __DPL_HIDE_FROM_ABI explicit constexpr direct_t() noexcept = default;
};

DPL_EXPORT inline constexpr direct_t direct{};

} // namespace datapar

namespace datapar::internal {
void to_simd_mask(...) noexcept = delete;

struct to_simd_mask_t {
private:
    template <typename T>
    requires is_enum_v<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool is_true(T val) noexcept {
        return is_true(__DPL to_underlying(val));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool is_true(signed_integral auto val) noexcept {
        return val < 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool is_true(unsigned_integral auto val) noexcept {
        return __DPL to_signed(val);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, FLATTEN, NODISCARD)
    static constexpr bool is_true(floating_point auto val) noexcept {
        using int_type = sbit_type_for_t<decltype(val)>;
        return is_true(__DPL bit_cast<int_type>(val));
    }

    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask<E, A> fallback(basic_simd<E, A> src) noexcept {
        return [&]<size_t... Is>(index_sequence<Is...>) {
            return dx::initialize<simd_mask<E, A>>(is_true(src[Is])...);
        }(iota_sequence<E, A>);
    }

public:
    template <basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr to_simd_mask_type_t<T> operator()(T src) noexcept {
        if constexpr (requires { to_simd_mask(internal::abi<T>, src); }) {
            if consteval {
                return fallback(src);
            } else {
                return to_simd_mask(internal::abi<T>, src);
            }
        } else {
            return fallback(src);
        }
    }

    template <simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr equivalent_mask_as<to_simd_mask_type_t<T>> auto operator()(
        T src) noexcept {
        if constexpr (requires { to_simd_mask(internal::abi<T>, src); }) {
            return to_simd_mask(internal::abi<T>, src);
        } else {
            return operator()(dx::to_basic_type(src));
        }
    }

    template <basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr to_simd_mask_type_t<T> operator()(
        direct_t tag, T src) noexcept {
        if constexpr (requires { to_simd_mask(internal::abi<T>, tag, src); }) {
            if consteval {
                return operator()(src);
            } else {
                return to_simd_mask(internal::abi<T>, tag, src);
            }
        } else {
            return operator()(src);
        }
    }

    template <simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr equivalent_mask_as<to_simd_mask_type_t<T>> auto operator()(
        direct_t tag, T src) noexcept {
        if constexpr (requires { to_simd_mask(internal::abi<T>, tag, src); }) {
            return to_simd_mask(internal::abi<T>, tag, src);
        } else {
            return operator()(tag, dx::to_basic_type(src));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::to_simd_mask_t to_simd_mask{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
