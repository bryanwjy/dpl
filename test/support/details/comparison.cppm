// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.support:comparison;
import :span;
import :data_generator;
import :linear_counter;
import :math_check;
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

template <double ULP>
struct ulp_cmp {
    // (exact - expected) / ulp(expected), per lane
    test::span<float const> residuals;

    explicit constexpr ulp_cmp(test::span<float const> r) noexcept
        : residuals(r) {}

    template <dpp::simd_vector V, dpl::same_as<V> R>
    constexpr auto operator()(V actual, R expected) const noexcept {
        using E = dpp::simd_element_type_t<V>;
        if constexpr (dpl::is_integral_v<E>) {
            return dpp::cmpeq(actual, expected);
        } else {
            auto const lanes = dpp::simd_abi_traits<V>::size();
            assert(residuals.size() == lanes);

            using A = dpp::simd_abi_type_t<V>;
            using sint_t = dpp::signed_representation_t<E>;
            using traits = dpl::floating_point_traits<E>;
            using IV = dpp::rebind_simd_t<V, sint_t>;
            auto const zero_i = dpp::broadcast<IV>(dpp::zero);

            // Residual sign per lane: +1, 0 or -1. NaN (non-finite expected)
            // gives 0; those lanes are decided by special_ok below.
            auto signs = make_array<sint_t, A>();
            for (auto const i : linear_counter(lanes))
                signs[i] = static_cast<sint_t>(residuals[i] > 0) -
                    static_cast<sint_t>(residuals[i] < 0);

            auto const rsign = dpp::load<IV>(signs.data());
            auto const r_ge0 = dpp::cmpge(rsign, zero_i);
            auto const r_le0 = dpp::cmple(rsign, zero_i);

            // Ordered key: monotonic in value, +0 and -0 both 0.
            auto const mag_mask = dpp::broadcast<IV>(
                dpl::to_signed(dpl::to_underlying(~traits::signbit)));
            auto const key = [&](V x) {
                auto const s = dpp::reinterpret<sint_t>(x);
                auto const mag = dpp::bwand(s, mag_mask);
                return dpp::select(
                    dpp::cmplt(s, zero_i), dpp::subtract(zero_i, mag), mag);
            };
            auto const d = dpp::subtract(key(actual), key(expected));

            auto const dist = [&] {
                auto dist = make_array<sint_t, A>();
                dpp::store(d, dist.data());
                return dist;
            }();

            auto pass = make_array<sint_t, A>();
            for (auto const i : linear_counter(lanes)) {
                auto const res = residuals[i];
                auto const err =
                    (res != res) ? 0.0f : static_cast<float>(dist[i]) - res;
                pass[i] = (err <= ULP && -err <= ULP) ? sint_t(-1) : sint_t(0);
            }
            auto const within = dpp::cmpneq(dpp::load<IV>(pass.data()), zero_i);
            auto const finite_ok = dpp::bwand(dpp::isfinite(actual), within);

            // Flush-to-zero: a subnormal expected value may come back as ±0.
            auto const min_normal = dpp::broadcast<IV>(
                dpl::to_signed(dpl::to_underlying(traits::mantissa_mask) + 1));
            auto const exp_mag =
                dpp::bwand(dpp::reinterpret<sint_t>(expected), mag_mask);
            auto const act_mag =
                dpp::bwand(dpp::reinterpret<sint_t>(actual), mag_mask);
            auto const flushed_ok = dpp::bwand(
                dpp::bwand(dpp::cmpgt(exp_mag, zero_i),
                    dpp::cmplt(exp_mag, min_normal)), // expected subnormal
                dpp::cmpeq(act_mag, zero_i));         // actual is ±0
            auto const value_ok = flush_to_zero<E>()
                ? dpp::bwor(finite_ok, flushed_ok)
                : finite_ok;

            auto const special_ok =
                dpp::bwor(dpp::cmpeq(dpp::reinterpret<sint_t>(actual),
                              dpp::reinterpret<sint_t>(expected)),
                    dpp::bwand(dpp::isnan(actual), dpp::isnan(expected)));

            return dpp::select(dpp::isfinite(expected), value_ok, special_ok);
        }
    }
};

} // namespace support
} // namespace dpl::test
