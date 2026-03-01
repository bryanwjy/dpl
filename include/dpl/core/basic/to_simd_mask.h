// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/to_basic_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/make_simd_mask_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT struct assume_cannonical_mask_t {
    __DPL_HIDE_FROM_ABI explicit constexpr assume_cannonical_mask_t() noexcept =
        default;
};

DPL_EXPORT inline constexpr assume_cannonical_mask_t assume_cannonical_mask{};
} // namespace datapar

namespace datapar::internal {
void to_simd_mask(...) noexcept = delete;

struct to_simd_mask_t {
public:
    template <basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr make_simd_mask_type_t<T> operator()(T src) noexcept
    requires requires { to_simd_mask(internal::abi<T>, src); }
    {
        return to_simd_mask(internal::abi<T>, src);
    }

    template <simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(T src) noexcept
        -> equivalent_mask_as<make_simd_mask_type_t<T>> auto {
        if constexpr (requires { to_simd_mask(internal::abi<T>, src); }) {
            return to_simd_mask(internal::abi<T>, src);
        } else {
            return operator()(dx::to_basic_type(src));
        }
    }

    template <basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr make_simd_mask_type_t<T> operator()(
        assume_cannonical_mask_t tag, T src) noexcept {
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
    static constexpr auto operator()(assume_cannonical_mask_t tag,
        T src) noexcept -> equivalent_mask_as<make_simd_mask_type_t<T>> auto {
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
