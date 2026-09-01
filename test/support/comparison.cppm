// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.test.support:comparison;
import :bitset_helpers;
import dpl;

export namespace dpl::test {

namespace dpp = dpl::datapar;
inline namespace support {
inline constexpr struct nancmp_t {
    static constexpr auto operator()(
        dpp::simd_vector auto lhs, dpp::simd_vector auto rhs) noexcept {
        return dpp::isnan(lhs) == dpp::isnan(rhs);
    }
} nancmp{};

inline constexpr struct bitcmp_t {
    template <dpp::simd_vector L, dpp::simd_vector R>
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using type = dpp::common_size_type_t<dpp::simd_element_type_t<L>,
            dpp::simd_element_type_t<R>>;
        using bits = dpp::unsigned_representation_t<type>;
        return dpp::reinterpret<bits>(lhs) == dpp::reinterpret<bits>(rhs);
    }

    template <typename L, typename R>
    requires (!dpp::simd_vector<L> && !dpp::simd_vector<R>) &&
        dpl::is_trivially_copyable_v<L> && dpl::is_trivially_copyable_v<R>
    static constexpr bool operator()(L lhs, R rhs) noexcept {
        return sizeof(L) == sizeof(R) &&
            dpl::to_bit_representation(lhs) == dpl::to_bit_representation(rhs);
    }
} bitcmp{};
} // namespace support
} // namespace dpl::test
