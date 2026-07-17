// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test;

// Tests for dpp::multiply on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::multiply(lhs, rhs)                  -- unmasked, always available
//   (2) dpp::multiply(src, mask, lhs, rhs)       -- merge-masked
//   (3) dpp::multiply(dpp::zero, mask, lhs, rhs) -- zero-masked explicit
//   (4) dpp::multiply(mask, lhs, rhs)            -- zero-masked alias (== form
//   3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types = dpl::type_pack<dpl::int8, dpl::uint8, dpl::int16, dpl::uint16,
        dpl::int32, dpl::uint32, dpl::int64, dpl::uint64, float, double,
        dpl::ext::float16, dpl::ext::bfloat16>;

    constexpr auto run_tests = []<typename E>(dpl::type_identity<E> type) {
        constexpr auto expected_op = [](E lhs, E rhs) { return lhs * rhs; };

        dpl::test::mt19937 engine(
            dpl::type_bit_v<E> % 31 + dpl::floating_point_like<E>);

        constexpr auto max = []() {
            if constexpr (dpl::integral<E>) {
                constexpr auto shift = dpl::type_bit_v<E> / 2;
                return dpl::integral_traits<E>::max_value >> shift;
            } else {
                constexpr auto bias =
                    dpl::floating_point_traits<E>::exponent_bias;
                constexpr auto exp =
                    dpl::floating_point_traits<E>::exponent_mask;
                constexpr auto shift = dpl::countr_zero(exp);
                // minus 1 for reduced range, so that sqrt * sqrt < max_value
                constexpr auto sqrt =
                    dpl::bitset<dpl::type_bit_v<E>>(
                        ((dpl::to_underlying(exp >> shift) - bias) >> 1) +
                        bias - 1)
                    << shift;
                return dpl::bit_cast<E>(sqrt);
            }
        }();

        constexpr auto min = [max]() {
            if constexpr (dpl::integral<E>) {
                constexpr auto shift = dpl::type_bit_v<E> / 2;
                return dpl::integral_traits<E>::min_value >> shift;
            } else {
                return -max;
            }
        }();

        dpl::test::array_generator<abi_t, E> const data_generator(min, max);
        dpl::test::scalar_generator<E> const src_generator(
            max * max, dpp::max_value_v<E>);

        auto const lhs = data_generator(engine);
        auto const rhs = data_generator(engine);
        auto const src = src_generator(engine);
        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op(lhs[i], rhs[i]);
        }
        dpl::test::binary_transform<abi_t>::test<E>(
            lhs, rhs, dpp::multiply, expected);
        dpl::test::binary_transform<abi_t>::test_masked<E>(
            lhs, rhs, dpp::multiply, src);

        auto const rnd = src_generator(engine);
        dpl::test::binary_transform<abi_t>::test<E>(rnd, 1, dpp::multiply, rnd);
        dpl::test::binary_transform<abi_t>::test<E>(1, rnd, dpp::multiply, rnd);
        dpl::test::binary_transform<abi_t>::test<E>(0, rnd, dpp::multiply, 0);
        dpl::test::binary_transform<abi_t>::test<E>(rnd, 0, dpp::multiply, 0);

        if constexpr (dpl::integral<E>) {
            if not consteval {
                constexpr auto int_min = dpl::integral_traits<E>::min_value;
                constexpr auto int_max = dpl::integral_traits<E>::max_value;
                dpl::test::binary_transform<abi_t>::test<E>(
                    int_max, 2, dpp::multiply, static_cast<E>(-2));

                if constexpr (dpl::signed_integral<E>) {
                    dpl::test::binary_transform<abi_t>::test<E>(
                        int_min, 2, dpp::multiply, 0);
                }

                dpl::test::binary_transform<abi_t>::test<E>(
                    int_max, int_max, dpp::multiply, 1);
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            dpl::test::binary_transform<abi_t>::test<E>(
                -0.0, 1.0, dpp::multiply, 0.0);
            dpl::test::binary_transform<abi_t>::test<E>(
                0.0, 1.0, dpp::multiply, 0.0);
            dpl::test::binary_transform<abi_t>::test<E>(
                -0.0, -0.0, dpp::multiply, 0.0);
            dpl::test::binary_transform<abi_t>::test<E>(
                1.0, -1.0, dpp::multiply, -1.0);

            if (dpl::test::ieee_denormal()) {
                using ubit_t = dpp::unsigned_representation_t<E>;
                auto const denorm = dpl::bit_cast<E>(ubit_t(1));
                dpl::test::binary_transform<abi_t>::test<E>(
                    denorm, denorm, dpp::multiply, 0.0);
                dpl::test::binary_transform<abi_t>::test<E>(
                    denorm, 2.0, dpp::multiply, dpl::bit_cast<E>(ubit_t(2)));
            }

            if (!dpl::test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                dpl::test::binary_transform<abi_t>::test<E>(
                    inf, 2.0, dpp::multiply, inf);
                dpl::test::binary_transform<abi_t>::test<E>(
                    inf, -2.0, dpp::multiply, -inf);
                dpl::test::binary_transform<abi_t>::test<E>(
                    -inf, 2.0, dpp::multiply, -inf);
                dpl::test::binary_transform<abi_t>::test<E>(
                    -inf, -2.0, dpp::multiply, inf);

                dpl::test::binary_transform<abi_t>::test<E>(
                    inf, inf, dpp::multiply, inf);
                dpl::test::binary_transform<abi_t>::test<E>(
                    inf, -inf, dpp::multiply, -inf);
                dpl::test::binary_transform<abi_t>::test<E>(
                    -inf, -inf, dpp::multiply, inf);

                dpl::test::binary_transform<abi_t>::test<E>(dpp::max_value_v<E>,
                    dpp::max_value_v<E>, dpp::multiply, inf);
                dpl::test::binary_transform<abi_t>::test<E>(dpp::max_value_v<E>,
                    -dpp::max_value_v<E>, dpp::multiply, -inf);
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
