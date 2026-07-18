// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;
import dpl.test;

// Tests for dpp::divide on xmm ABI. Requires SSE4.2 (implied by dpl.xmm).
//
// Forms tested:
//   (1) dpp::divide(lhs, rhs)                  -- unmasked, always available
//   (2) dpp::divide(src, mask, lhs, rhs)       -- merge-masked
//   (3) dpp::divide(dpp::zero, mask, lhs, rhs) -- zero-masked explicit
//   (4) dpp::divide(mask, lhs, rhs)            -- zero-masked alias (== form
//   3)

int main() {
    namespace dpp = dpl::datapar;
    namespace xmm = dpl::datapar::xmm;
    using abi_t = xmm::abi_tag;

    using types =
        dpl::type_pack<float, double, dpl::ext::float16, dpl::ext::bfloat16>;

    constexpr auto run_tests = []<typename E>(dpl::type_identity<E> type) {
        constexpr auto expected_op = [](E lhs, E rhs) { return lhs / rhs; };

        dpl::test::mt19937 engine(
            (dpl::type_bit_v<E> + dpl::floating_point_traits<E>::digits) % 17);

        constexpr auto max = []() {
            constexpr auto bias = dpl::floating_point_traits<E>::exponent_bias;
            constexpr auto exp = dpl::floating_point_traits<E>::exponent_mask;
            constexpr auto shift = dpl::countr_zero(exp);
            // minus 1 for reduced range, so that sqrt * sqrt < max_value
            constexpr auto sqrt =
                dpl::bitset<dpl::type_bit_v<E>>(
                    ((dpl::to_underlying(exp >> shift) - bias) >> 1) + bias - 1)
                << shift;
            return dpl::bit_cast<E>(sqrt);
        }();

        constexpr auto min = -max;

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
            lhs, rhs, dpp::divide, expected);
        dpl::test::binary_transform<abi_t>::test_masked<E>(
            lhs, rhs, dpp::divide, src);

        auto const rnd = src_generator(engine);
        dpl::test::binary_transform<abi_t>::test<E>(0, rnd, dpp::divide, 0);
        dpl::test::binary_transform<abi_t>::test<E>(rnd, 1, dpp::divide, rnd);
        dpl::test::binary_transform<abi_t>::test<E>(rnd, -1, dpp::divide, -rnd);

        if constexpr (dpl::floating_point_like<E>) {

            if (dpl::test::ieee_denormal()) {
                using ubit_t = dpp::unsigned_representation_t<E>;
                auto const denorm = dpl::bit_cast<E>(ubit_t(1));
                dpl::test::binary_transform<abi_t>::test<E>(
                    1, denorm, dpp::divide, dpp::infinity_v<E>);
                dpl::test::binary_transform<abi_t>::test<E>(
                    dpl::bit_cast<E>(ubit_t(2)), 2.0, dpp::divide, denorm);
            }

            if (!dpl::test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;

                dpl::test::binary_transform<abi_t>::test<E>(
                    inf, 2.0, dpp::divide, inf);
                dpl::test::binary_transform<abi_t>::test<E>(
                    inf, -2.0, dpp::divide, -inf);
                dpl::test::binary_transform<abi_t>::test<E>(
                    -inf, 2.0, dpp::divide, -inf);
                dpl::test::binary_transform<abi_t>::test<E>(
                    -inf, -2.0, dpp::divide, inf);

                dpl::test::binary_transform<abi_t>::test<E>(
                    2.0, inf, dpp::divide, 0.0, dpl::test::bitcmp);
                dpl::test::binary_transform<abi_t>::test<E>(
                    2.0, -inf, dpp::divide, -0.0, dpl::test::bitcmp);
                dpl::test::binary_transform<abi_t>::test<E>(
                    -2.0, inf, dpp::divide, -0.0, dpl::test::bitcmp);
                dpl::test::binary_transform<abi_t>::test<E>(
                    -2.0, -inf, dpp::divide, 0.0, dpl::test::bitcmp);

                if not consteval {
                    dpl::test::binary_transform<abi_t>::test<E>(
                        1, 0, dpp::divide, inf);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        -1, 0, dpp::divide, -inf);

                    auto const cmpunord = [](auto lhs, auto rhs) {
                        return dpp::isnan(lhs) && dpp::isnan(rhs);
                    };
                    dpl::test::binary_transform<abi_t>::test<E>(
                        inf, inf, dpp::divide, dpp::nan, cmpunord);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        inf, -inf, dpp::divide, dpp::nan, cmpunord);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        -inf, inf, dpp::divide, dpp::nan, cmpunord);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        -inf, -inf, dpp::divide, dpp::nan, cmpunord);

                    dpl::test::binary_transform<abi_t>::test<E>(
                        0.0, 0.0, dpp::divide, dpp::nan, cmpunord);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        -0.0, 0.0, dpp::divide, dpp::nan, cmpunord);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        0.0, -0.0, dpp::divide, dpp::nan, cmpunord);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        -0.0, -0.0, dpp::divide, dpp::nan, cmpunord);

                    dpl::test::binary_transform<abi_t>::test<E>(
                        dpp::nan, 1, dpp::divide, dpp::nan, cmpunord);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        1, dpp::nan, dpp::divide, dpp::nan, cmpunord);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        inf, dpp::nan, dpp::divide, dpp::nan, cmpunord);
                    dpl::test::binary_transform<abi_t>::test<E>(
                        dpp::nan, inf, dpp::divide, dpp::nan, cmpunord);
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
