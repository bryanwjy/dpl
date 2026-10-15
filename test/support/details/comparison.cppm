// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.test.support:comparison;
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
        return dpp::cmpeq(
            dpp::reinterpret<bits>(lhs), dpp::reinterpret<bits>(rhs));
    }

    template <typename L, typename R>
    requires (!dpp::simd_vector<L> && !dpp::simd_vector<R>) &&
        dpl::is_trivially_copyable_v<L> && dpl::is_trivially_copyable_v<R>
    static constexpr bool operator()(L lhs, R rhs) noexcept {
        return sizeof(L) == sizeof(R) &&
            dpl::to_bit_representation(lhs) == dpl::to_bit_representation(rhs);
    }
} bitcmp{};

template <size_t N>
struct precision_cmp_t {
    template <dpp::simd_vector L, dpl::same_as<L> R>
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using E = dpp::simd_element_type_t<L>;
        if constexpr (dpl::is_integral_v<E>) {
            return dpp::cmpeq(lhs, rhs);
        } else {
            static_assert(N <
                dpl::popcount(dpl::floating_point_traits<E>::mantissa_mask));
            using sint_t = dpp::signed_representation_t<E>;
            auto const diff =
                dpp::reinterpret<sint_t>(lhs) - dpp::reinterpret<sint_t>(rhs);
            return dpp::cmplt(dpp::abs(diff), dpl::to_signed(N));
        }
    }

    template <typename L, common_with<L> R>
    requires (!dpp::simd_vector<L> && !dpp::simd_vector<R>) &&
        dpl::is_trivially_copyable_v<L> && dpl::is_trivially_copyable_v<R>
    static constexpr bool operator()(L lhs, R rhs) noexcept {
        using E = common_type_t<L, R>;
        static_assert(dpl::is_integral_v<E> || dpl::floating_point_like<E>);
        if constexpr (dpl::is_integral_v<E>) {
            return lhs == rhs;
        } else {
            static_assert(N <
                dpl::popcount(dpl::floating_point_traits<E>::mantissa_mask));
            using sint_t = dpp::signed_representation_t<E>;
            auto diff = dpl::bit_cast<sint_t>(lhs) - dpl::bit_cast<sint_t>(rhs);
            diff = diff < 0 ? -diff : diff;
            return diff < dpl::to_signed(N);
        }
    }
};

template <size_t N>
inline constexpr precision_cmp_t<N> precision_cmp{};

} // namespace support
} // namespace dpl::test
