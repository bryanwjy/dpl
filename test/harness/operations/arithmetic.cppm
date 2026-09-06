// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.arithmetic;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_primitive_operation auto arop, dpp::simd_abi A>
class arithmetic {
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept
    requires (arop == dpp::add)
    {
        if constexpr (dpl::signed_integral<E>) {
            return dpl::to_signed(
                dpl::to_unsigned(lhs) + dpl::to_unsigned(rhs));
        } else {
            return lhs + rhs;
        }
    }

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept
    requires (arop == dpp::subtract)
    {
        if constexpr (dpl::signed_integral<E>) {
            return dpl::to_signed(
                dpl::to_unsigned(lhs) - dpl::to_unsigned(rhs));
        } else {
            return lhs - rhs;
        }
    }

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept
    requires (arop == dpp::multiply)
    {
        return lhs * rhs;
    }

    template <typename E>
    static constexpr E expected_op(E lhs, E rhs) noexcept
    requires (arop == dpp::divide)
    {
        return lhs / rhs;
    }

    template <typename E>
    struct generators_t {
        static constexpr E sqrtmax = []() {
            if constexpr (dpl::integral<E>) {
                constexpr auto shift = dpl::type_bit_v<E> / 2;
                return (dpl::integral_traits<E>::max_value >> shift) - 1;
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

        static constexpr E sqrtmin = []() {
            if constexpr (dpl::integral<E>) {
                constexpr auto shift = dpl::type_bit_v<E> / 2;
                return (dpl::integral_traits<E>::min_value >> shift) + 1;
            } else {
                return -sqrtmax;
            }
        }();

        constexpr generators_t() noexcept
            : data(sqrtmin, sqrtmax)
            , src(sqrtmax * sqrtmax, dpp::max_value_v<E>) {}

        test::array_generator<A, E> data;
        test::scalar_generator<E> src;
    };

    template <typename E, rng_like Rng>
    static constexpr void operation_specific_tests(
        generators_t<E> const& generators, Rng& engine)
    requires (arop == dpp::add)
    {
        if constexpr (dpl::integral<E>) {
            test::binary_transform<A>::template test<E>(0, 0, dpp::add, 0);
            auto const rnd = generators.src(engine);
            test::binary_transform<A>::template test<E>(rnd, 0, dpp::add, rnd);
            test::binary_transform<A>::template test<E>(0, rnd, dpp::add, rnd);

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
                test::binary_transform<A>::template test<E>(
                    max, 1, dpp::add, min);
                test::binary_transform<A>::template test<E>(
                    min, -1, dpp::add, max);
                test::binary_transform<A>::template test<E>(
                    max, max, dpp::add, max - 1);

                test::binary_transform<A>::template test<E>(
                    min, max, dpp::add, static_cast<E>(-1));
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            test::binary_transform<A>::template test<E>(
                -0.0, 0.0, dpp::add, 0.0);
            test::binary_transform<A>::template test<E>(
                0.0, 0.0, dpp::add, 0.0);
            test::binary_transform<A>::template test<E>(
                -0.0, -0.0, dpp::add, 0.0);
            test::binary_transform<A>::template test<E>(
                1.0, -1.0, dpp::add, 0.0);

            if (!test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                test::binary_transform<A>::template test<E>(
                    inf, 1.0, dpp::add, inf);
                test::binary_transform<A>::template test<E>(
                    -inf, -1.0, dpp::add, -inf);
                if not consteval {
                    // nan producing arithmetic is not allowed at constexpr
                    test::binary_transform<A>::template test<E>(inf, -inf,
                        dpp::add, dpp::nan_v<E>, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                    test::binary_transform<A>::template test<E>(dpp::nan, 1.0,
                        dpp::add, dpp::nan, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });

                    test::binary_transform<A>::template test<E>(1.0, dpp::nan,
                        dpp::add, dpp::nan, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                }
            }
        }
    }

    template <typename E, rng_like Rng>
    static constexpr void operation_specific_tests(
        generators_t<E> const& generators, Rng& engine)
    requires (arop == dpp::subtract)
    {
        if constexpr (dpl::integral<E>) {
            dpl::test::binary_transform<A>::template test<E>(
                0, 0, dpp::subtract, 0);
            auto const rnd = generators.src(engine);
            dpl::test::binary_transform<A>::template test<E>(
                rnd, 0, dpp::subtract, rnd);
            dpl::test::binary_transform<A>::template test<E>(
                0, rnd, dpp::subtract, -rnd);
            dpl::test::binary_transform<A>::template test<E>(
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
                dpl::test::binary_transform<A>::template test<E>(
                    min, 1, dpp::subtract, max);
                dpl::test::binary_transform<A>::template test<E>(
                    max, -1, dpp::subtract, min);
                dpl::test::binary_transform<A>::template test<E>(
                    min, max, dpp::subtract, 1);
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            dpl::test::binary_transform<A>::template test<E>(
                -0.0, 0.0, dpp::subtract, 0.0);
            dpl::test::binary_transform<A>::template test<E>(
                0.0, 0.0, dpp::subtract, 0.0);
            dpl::test::binary_transform<A>::template test<E>(
                -0.0, -0.0, dpp::subtract, 0.0);
            dpl::test::binary_transform<A>::template test<E>(
                1.0, 1.0, dpp::subtract, 0.0);
            dpl::test::binary_transform<A>::template test<E>(
                -1.0, -1.0, dpp::subtract, 0.0);

            if (!dpl::test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                dpl::test::binary_transform<A>::template test<E>(
                    inf, 1.0, dpp::subtract, inf);
                dpl::test::binary_transform<A>::template test<E>(
                    -inf, -1.0, dpp::subtract, -inf);
                if not consteval {
                    // nan producing arithmetic is not allowed at constexpr
                    dpl::test::binary_transform<A>::template test<E>(-inf, -inf,
                        dpp::subtract, dpp::nan_v<E>, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                    dpl::test::binary_transform<A>::template test<E>(inf, inf,
                        dpp::subtract, dpp::nan_v<E>, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                    dpl::test::binary_transform<A>::template test<E>(dpp::nan,
                        1.0, dpp::subtract, dpp::nan, [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });

                    dpl::test::binary_transform<A>::template test<E>(1.0,
                        dpp::nan, dpp::subtract, dpp::nan,
                        [](auto lhs, auto rhs) {
                            return dpp::isnan(lhs) && dpp::isnan(rhs);
                        });
                }
            }
        }
    }

    template <typename E, rng_like Rng>
    static constexpr void operation_specific_tests(
        generators_t<E> const& generators, Rng& engine)
    requires (arop == dpp::multiply)
    {
        auto const rnd = generators.src(engine);
        test::binary_transform<A>::template test<E>(rnd, 1, dpp::multiply, rnd);
        test::binary_transform<A>::template test<E>(1, rnd, dpp::multiply, rnd);
        test::binary_transform<A>::template test<E>(0, rnd, dpp::multiply, 0);
        test::binary_transform<A>::template test<E>(rnd, 0, dpp::multiply, 0);

        if constexpr (dpl::integral<E>) {
            if not consteval {
                constexpr auto int_min = dpl::integral_traits<E>::min_value;
                constexpr auto int_max = dpl::integral_traits<E>::max_value;
                test::binary_transform<A>::template test<E>(
                    int_max, 2, dpp::multiply, static_cast<E>(-2));

                if constexpr (dpl::signed_integral<E>) {
                    test::binary_transform<A>::template test<E>(
                        int_min, 2, dpp::multiply, 0);
                }

                test::binary_transform<A>::template test<E>(
                    int_max, int_max, dpp::multiply, 1);
            }
        }

        if constexpr (dpl::floating_point_like<E>) {
            test::binary_transform<A>::template test<E>(
                -0.0, 1.0, dpp::multiply, 0.0);
            test::binary_transform<A>::template test<E>(
                0.0, 1.0, dpp::multiply, 0.0);
            test::binary_transform<A>::template test<E>(
                -0.0, -0.0, dpp::multiply, 0.0);
            test::binary_transform<A>::template test<E>(
                1.0, -1.0, dpp::multiply, -1.0);

            if (test::ieee_denormal()) {
                using ubit_t = dpp::unsigned_representation_t<E>;
                auto const denorm = dpl::bit_cast<E>(ubit_t(1));
                test::binary_transform<A>::template test<E>(
                    denorm, denorm, dpp::multiply, 0.0);
                test::binary_transform<A>::template test<E>(
                    denorm, 2.0, dpp::multiply, dpl::bit_cast<E>(ubit_t(2)));
            }

            if (!test::finite_math_only()) {
                constexpr auto inf = dpp::infinity_v<E>;
                test::binary_transform<A>::template test<E>(
                    inf, 2.0, dpp::multiply, inf);
                test::binary_transform<A>::template test<E>(
                    inf, -2.0, dpp::multiply, -inf);
                test::binary_transform<A>::template test<E>(
                    -inf, 2.0, dpp::multiply, -inf);
                test::binary_transform<A>::template test<E>(
                    -inf, -2.0, dpp::multiply, inf);

                test::binary_transform<A>::template test<E>(
                    inf, inf, dpp::multiply, inf);
                test::binary_transform<A>::template test<E>(
                    inf, -inf, dpp::multiply, -inf);
                test::binary_transform<A>::template test<E>(
                    -inf, -inf, dpp::multiply, inf);

                test::binary_transform<A>::template test<E>(dpp::max_value_v<E>,
                    dpp::max_value_v<E>, dpp::multiply, inf);
                test::binary_transform<A>::template test<E>(dpp::max_value_v<E>,
                    -dpp::max_value_v<E>, dpp::multiply, -inf);
            }
        }
    }

    template <typename E, rng_like Rng>
    static constexpr void operation_specific_tests(
        generators_t<E> const& generators, Rng& engine)
    requires (arop == dpp::divide)
    {
        auto const rnd = generators.src(engine);
        test::binary_transform<A>::template test<E>(0, rnd, dpp::divide, 0);
        test::binary_transform<A>::template test<E>(rnd, 1, dpp::divide, rnd);
        test::binary_transform<A>::template test<E>(rnd, -1, dpp::divide, -rnd);

        if (test::ieee_denormal()) {
            using ubit_t = dpp::unsigned_representation_t<E>;
            auto const denorm = dpl::bit_cast<E>(ubit_t(1));
            test::binary_transform<A>::template test<E>(
                1, denorm, dpp::divide, dpp::infinity_v<E>);
            test::binary_transform<A>::template test<E>(
                dpl::bit_cast<E>(ubit_t(2)), 2.0, dpp::divide, denorm);
        }

        if (!test::finite_math_only()) {
            constexpr auto inf = dpp::infinity_v<E>;

            test::binary_transform<A>::template test<E>(
                inf, 2.0, dpp::divide, inf);
            test::binary_transform<A>::template test<E>(
                inf, -2.0, dpp::divide, -inf);
            test::binary_transform<A>::template test<E>(
                -inf, 2.0, dpp::divide, -inf);
            test::binary_transform<A>::template test<E>(
                -inf, -2.0, dpp::divide, inf);

            test::binary_transform<A>::template test<E>(
                2.0, inf, dpp::divide, 0.0, test::bitcmp);
            test::binary_transform<A>::template test<E>(
                2.0, -inf, dpp::divide, -0.0, test::bitcmp);
            test::binary_transform<A>::template test<E>(
                -2.0, inf, dpp::divide, -0.0, test::bitcmp);
            test::binary_transform<A>::template test<E>(
                -2.0, -inf, dpp::divide, 0.0, test::bitcmp);

            if not consteval {
                test::binary_transform<A>::template test<E>(
                    1, 0, dpp::divide, inf);
                test::binary_transform<A>::template test<E>(
                    -1, 0, dpp::divide, -inf);

                auto const cmpunord = [](auto lhs, auto rhs) {
                    return dpp::isnan(lhs) && dpp::isnan(rhs);
                };
                test::binary_transform<A>::template test<E>(
                    inf, inf, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<A>::template test<E>(
                    inf, -inf, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<A>::template test<E>(
                    -inf, inf, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<A>::template test<E>(
                    -inf, -inf, dpp::divide, dpp::nan, cmpunord);

                test::binary_transform<A>::template test<E>(
                    0.0, 0.0, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<A>::template test<E>(
                    -0.0, 0.0, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<A>::template test<E>(
                    0.0, -0.0, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<A>::template test<E>(
                    -0.0, -0.0, dpp::divide, dpp::nan, cmpunord);

                test::binary_transform<A>::template test<E>(
                    dpp::nan, 1, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<A>::template test<E>(
                    1, dpp::nan, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<A>::template test<E>(
                    inf, dpp::nan, dpp::divide, dpp::nan, cmpunord);
                test::binary_transform<A>::template test<E>(
                    dpp::nan, inf, dpp::divide, dpp::nan, cmpunord);
            }
        }
    }

    template <typename E>
    static constexpr auto mixmask() noexcept
    requires (arop == dpp::addsub)
    {
        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto lanes = abi_traits<E>::size();
            using bitset_t = dpl::bitset<lanes>;
            using alt_cmask_t =
                dpp::const_mask<lanes, test::repeat_byte<lanes>(0x55u)>;
            return alt_cmask_t();
        } else {
            return dpp::cmpeq(
                dpp::bwand(dpp::lane_index<A, E>(), dpp::one), dpp::zero);
        }
    }

    template <typename E>
    static constexpr auto mixmask() noexcept
    requires (arop == dpp::subadd)
    {
        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto lanes = abi_traits<E>::size();
            using bitset_t = dpl::bitset<lanes>;
            using alt_cmask_t =
                dpp::const_mask<lanes, test::repeat_byte<lanes>(0xAAu)>;
            return alt_cmask_t();
        } else {
            return dpp::cmpeq(
                dpp::bwand(dpp::lane_index<A, E>(), dpp::one), dpp::one);
        }
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return arithmetic::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        generators_t<E> const generators;
        {
            auto const lhs = generators.data(engine);
            auto const rhs = generators.data(engine);
            auto expected = lhs;
            for (auto i = 0zu; i < expected.size(); ++i) {
                expected[i] = expected_op(lhs[i], rhs[i]);
            }

            test::binary_transform<A>::template test<E>(
                lhs, rhs, arop, expected, test::bitcmp);
        }

        {
            auto lhs = generators.data(engine);
            auto rhs = generators.data(engine);
            if constexpr (dpl::unsigned_integral<E> && dpp::subtract == arop) {
                // prevent wraparound from intersecting with src
                for (auto i = 0zu; i < rhs.size(); ++i) {
                    if (lhs[i] < rhs[i]) {
                        dpl::ranges::swap(lhs[i], rhs[i]);
                    }
                }
            }
            auto const src = generators.src(engine);
            test::binary_transform<A>::template test_masked<E>(
                lhs, rhs, arop, src);
        }

        operation_specific_tests<E>(generators, engine);

        return true;
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires (arop == dpp::addsub || arop == dpp::subadd)
    {
        generators_t<E> const generators;
        constexpr auto expected_op = [](vec_t<E> vlhs, vec_t<E> vrhs) noexcept {
            return dpp::select(
                mixmask<E>(), dpp::add(vlhs, vrhs), dpp::subtract(vlhs, vrhs));
        };
        for (auto i = 0zu; i < 4; ++i) {
            {
                auto const lhs = generators.data(engine);
                auto const rhs = generators.data(engine);
                auto const vlhs = dpp::load<E, A>(lhs.data());
                auto const vrhs = dpp::load<E, A>(rhs.data());
                auto const vexpected = expected_op(vlhs, vrhs);
                auto const vactual = arop(vlhs, vrhs);

                assert(dpp::all_of(vexpected == vactual));
            }

            {
                auto lhs = generators.data(engine);
                auto rhs = generators.data(engine);
                if constexpr (dpl::unsigned_integral<E>) {
                    // prevent wraparound from intersecting with src
                    for (auto i = 0zu; i < rhs.size(); ++i) {
                        if (lhs[i] < rhs[i]) {
                            dpl::ranges::swap(lhs[i], rhs[i]);
                        }
                    }
                }

                auto const src = generators.src(engine);
                dpl::test::binary_transform<A>::template test_masked<E>(
                    lhs, rhs, arop, src);
            }
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using addition = arithmetic<dpp::add, A>;
export template <dpp::simd_abi A>
using subtraction = arithmetic<dpp::subtract, A>;
export template <dpp::simd_abi A>
using multiplication = arithmetic<dpp::multiply, A>;
export template <dpp::simd_abi A>
using division = arithmetic<dpp::divide, A>;
export template <dpp::simd_abi A>
using addsubtraction = arithmetic<dpp::addsub, A>;
export template <dpp::simd_abi A>
using subaddition = arithmetic<dpp::subadd, A>;
} // namespace dpl::test
