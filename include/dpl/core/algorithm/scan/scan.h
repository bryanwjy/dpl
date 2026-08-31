// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/common.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void scan(...) noexcept = delete;

struct scan_t : public inclusive_scan_base<scan_t> {
    using operation_base<scan_t>::operator();
};

template <>
struct operation_signature<scan_t> {
    template <simd_vector T, scan_operator_for<T> Op>
    static consteval void operator()(T&&, Op&&) noexcept {}
};

template <typename T, typename Op>
concept unqualified_canonical_scan = requires {
    {
        scan(internal::abi<T>, internal::declarg<T>(), internal::declarg<Op>())
    } -> same_as<T>;
};

template <>
struct canonical_impl<scan_t> {
    template <canonical_vector T, scan_operator_for<T> Op>
    requires unqualified_canonical_scan<T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Op&& func) noexcept {
        return scan(internal::abi<T>, val, __DPL forward<Op>(func));
    }
};

template <typename T, typename Op>
concept unqualified_extended_scan = requires {
    {
        scan(internal::declarg<T>(), internal::declarg<Op>())
    } -> equivalent_vector_with<T>;
};

template <>
struct extended_impl<scan_t> {
public:
    template <extended_vector T, scan_operator_for<T> Op>
    requires unqualified_extended_scan<T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Op&& func) {
        return scan(__DPL forward<T>(val), __DPL forward<Op>(func));
    }
};

template <>
struct fallback_impl<scan_t> {
public:
    template <simd_vector T, scan_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Op&& func) {
        return internal::inclusive_scan(
            auto(__DPL forward<T>(val)), __DPL forward<Op>(func));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::scan_t scan{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
