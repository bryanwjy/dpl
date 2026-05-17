// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/operation_category.h"
#include "dpl/core/concepts/simd_abi_traits.h"
#include "dpl/core/concepts/simd_type.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/conditional.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename>
struct simd_traits {};

DPL_EXPORT template <simd_class T>
struct simd_traits<T> {
    using type = T;
    using abi_type = typename T::abi_type;
    using value_type = typename T::value_type;
    using element_type = simd_lane_type_t<T>;
    using element_representation = simd_lane_representation_t<T>;
    using native_type = conditional_t<simd_mask_type<T>,
        typename simd_abi_traits<abi_type, element_type>::native_mask,
        typename simd_abi_traits<abi_type, element_type>::native_type>;
    static constexpr operation_category decay_policy = []() {
        if constexpr (basic_simd_class<T>) {
            return T::all;
        } else if constexpr (requires { T::decay_policy; }) {
            static_assert(requires {
                typename integral_constant<operation_category, T::decay_policy>;
            });

            return T::decay_policy;
        } else {
            return operation_category::lane_agnostic;
        }
    }();
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
