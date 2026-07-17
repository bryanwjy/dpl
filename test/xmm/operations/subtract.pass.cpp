// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test;

// Tests for dpp::subtract on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::subtract(lhs, rhs)                  -- unmasked, always available
//   (2) dpp::subtract(src, mask, lhs, rhs)       -- merge-masked
//   (3) dpp::subtract(dpp::zero, mask, lhs, rhs) -- zero-masked explicit
//   (4) dpp::subtract(mask, lhs, rhs)            -- zero-masked alias (== form
//   3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types = dpl::type_pack<dpl::int8, dpl::uint8, dpl::int16, dpl::uint16,
        dpl::int32, dpl::uint32, dpl::int64, dpl::uint64, float, double,
        dpl::ext::float16, dpl::ext::bfloat16>;

    constexpr auto run_tests = []<typename E>(dpl::type_identity<E> type) {
        constexpr auto expected_op = [](E lhs, E rhs) {
            if constexpr (dpl::signed_integral<E>) {
                return dpl::to_signed(
                    dpl::to_unsigned(lhs) - dpl::to_unsigned(rhs));
            } else {
                return lhs - rhs;
            }
        };

        dpl::test::mt19937 engine(
            dpl::type_bit_v<E> % 31 + dpl::floating_point_like<E>);

        dpl::test::array_generator<abi_t, E> const data_generator(
            dpl::test::half_range);
        dpl::test::scalar_generator<E> const src_generator(
            dpp::max_value_v<E> / 4 * 3, dpp::max_value_v<E>);

        auto const lhs = data_generator(engine);
        auto const rhs = data_generator(engine);
        auto const src = src_generator(engine);
        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op(lhs[i], rhs[i]);
        }
        dpl::test::binary_transform<abi_t>::test<E>(
            lhs, rhs, dpp::subtract, expected);
        dpl::test::binary_transform<abi_t>::test_masked<E>(
            lhs, rhs, dpp::subtract, src);

        if constexpr (dpl::integral<E>) {
            dpl::test::binary_transform<abi_t>::test<E>(0, 0, dpp::subtract, 0);
            auto const rnd = src_generator(engine);
            dpl::test::binary_transform<abi_t>::test<E>(
                rnd, 0, dpp::subtract, rnd);
            dpl::test::binary_transform<abi_t>::test<E>(
                0, rnd, dpp::subtract, -rnd);
            dpl::test::binary_transform<abi_t>::test<E>(
                rnd, rnd, dpp::subtract, 0);

            constexpr auto max = dpl::integral_traits<E>::max_value;
            constexpr auto min = dpl::integral_traits<E>::min_value;

            auto const wraparound_test = []() {
                if constexpr (dpl::unsigned_integral<E>) {
                    return true;
                } else if not consteval {
                    return true;
                } else {
                    return false;
                }
            }();
            if (wraparound_test) {
                dpl::test::binary_transform<abi_t>::test<E>(
                    min, 1, dpp::subtract, max);
                dpl::test::binary_transform<abi_t>::test<E>(
                    max, -1, dpp::subtract, min);
                dpl::test::binary_transform<abi_t>::test<E>(
                    min, max, dpp::subtract, 1);
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            dpl::test::binary_transform<abi_t>::test<E>(
                -0.0, 0.0, dpp::subtract, 0.0);
            dpl::test::binary_transform<abi_t>::test<E>(
                0.0, 0.0, dpp::subtract, 0.0);
            dpl::test::binary_transform<abi_t>::test<E>(
                -0.0, -0.0, dpp::subtract, 0.0);
            dpl::test::binary_transform<abi_t>::test<E>(
                1.0, 1.0, dpp::subtract, 0.0);
            dpl::test::binary_transform<abi_t>::test<E>(
                -1.0, -1.0, dpp::subtract, 0.0);

            if (!dpl::test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                dpl::test::binary_transform<abi_t>::test<E>(
                    inf, 1.0, dpp::subtract, inf);
                dpl::test::binary_transform<abi_t>::test<E>(
                    -inf, -1.0, dpp::subtract, -inf);
                if not consteval {
                    // nan producing arithmetic is not allowed at constexpr
                    dpl::test::binary_transform<abi_t>::test<E>(-inf, -inf,
                        dpp::subtract, dpp::nan_v<E>, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                    dpl::test::binary_transform<abi_t>::test<E>(inf, inf,
                        dpp::subtract, dpp::nan_v<E>, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                    dpl::test::binary_transform<abi_t>::test<E>(dpp::nan, 1.0,
                        dpp::subtract, dpp::nan, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });

                    dpl::test::binary_transform<abi_t>::test<E>(1.0, dpp::nan,
                        dpp::subtract, dpp::nan, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                }
            }
        }

        return true;
    };

    dpl::pack::for_each(
        [=](auto tp) {
            static_assert(run_tests(tp));
            assert(run_tests(tp));
        },
        types{});
    return 0;
}
