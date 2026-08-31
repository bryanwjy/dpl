// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/rotate.h"
#include "dpl/core/algorithm/scan/common.h"
#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void exscan(...) noexcept = delete;

struct exscan_t : public exclusive_scan_base<exscan_t> {
    using operation_base<exscan_t>::operator();
};

template <>
struct operation_signature<exscan_t> {
    template <simd_vector T, broadcastable_to<T> V, scan_operator_for<T> Op>
    static consteval void operator()(T&&, V&&, Op&&) noexcept {}
};

template <typename T, typename V, typename Op>
concept unqualified_canonical_exscan = requires {
    {
        exscan(internal::abi<T>, internal::declarg<T>(), internal::declarg<V>(),
            internal::declarg<Op>())
    } -> same_as<T>;
};

template <>
struct canonical_impl<exscan_t> {
    template <canonical_vector T, broadcastable_to<T> V,
        scan_operator_for<T> Op>
    requires unqualified_canonical_exscan<T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, V&& init, Op&& func) noexcept {
        return exscan(internal::abi<T>, val, __DPL forward<V>(init),
            __DPL forward<Op>(func));
    }
};

template <typename T, typename V, typename Op>
concept unqualified_extended_exscan = requires {
    {
        exscan(internal::declarg<T>(), internal::declarg<V>(),
            internal::declarg<Op>())
    } -> equivalent_vector_with<T>;
};

template <>
struct extended_impl<exscan_t> {
public:
    template <extended_vector T, broadcastable_to<T> V, scan_operator_for<T> Op>
    requires unqualified_extended_exscan<T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, V&& init, Op&& func) {
        return exscan(__DPL forward<T>(val), __DPL forward<V>(init),
            __DPL forward<Op>(func));
    }
};

template <>
struct fallback_impl<exscan_t> {
    template <simd_vector T, broadcastable_to<T> V, scan_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, V&& init, Op&& func) noexcept {
        auto slid = dx::slide_right(dx::broadcast<T>(__DPL forward<V>(init)),
            __DPL forward<T>(val), imm<1zu>);
        return internal::inclusive_scan(
            __DPL move(slid), __DPL forward<Op>(func));
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::exscan_t exscan{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
