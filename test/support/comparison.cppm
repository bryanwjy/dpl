// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.test:support.comparison;

import dpl;

export namespace dpl::test {

namespace dpp = dpl::datapar;

inline constexpr struct nancmp_t {
    static constexpr auto operator()(
        dpp::simd_vector auto lhs, dpp::simd_vector auto rhs) noexcept {
        return dpp::isnan(lhs) == dpp::isnan(rhs);
    }
} nancmp{};

inline constexpr struct bitcmp_t {
    template <dpp::simd_vector T>
    static constexpr auto operator()(T lhs, T rhs) noexcept {
        using type = typename T::value_type;
        return dpp::reinterpret<dpp::signed_representation_t<type>>(lhs) ==
            dpp::reinterpret<dpp::signed_representation_t<type>>(rhs);
    }

    template <typename T>
    requires dpl::is_scalar_v<T>
    static constexpr auto operator()(T lhs, T rhs) noexcept {
        return dpl::bit_cast<dpp::unsigned_representation_t<T>>(lhs) ==
            dpl::bit_cast<dpp::unsigned_representation_t<T>>(rhs);
    }
} bitcmp{};

} // namespace dpl::test
