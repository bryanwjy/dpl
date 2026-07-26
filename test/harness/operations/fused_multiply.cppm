// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.fused_multiply;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_primitive_operation auto fmop, dpp::simd_abi A>
class fused_multiply {
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr E expected_op(E lhs, E mid, E rhs) noexcept {
        if constexpr (fmop == dpp::fmadd) {
            return lhs * mid + rhs;
        } else if constexpr (fmop == dpp::fmsub) {
            return lhs * mid - rhs;
        } else if constexpr (fmop == dpp::fnmadd) {
            return rhs - lhs * mid;
        } else if constexpr (fmop == dpp::fnmsub) {
            return -rhs - lhs * mid;
        } else if constexpr (fmop == dpp::fmacc) {
            return lhs + mid * rhs;
        } else if constexpr (fmop == dpp::fmsac) {
            return mid * rhs - lhs;
        } else if constexpr (fmop == dpp::fnmacc) {
            return lhs - mid * rhs;
        } else {
            static_assert(fmop == dpp::fnmsac);
            return -lhs - mid * rhs;
        }
    }

    template <typename E>
    static constexpr auto max = []() {
        auto bias = dpl::floating_point_traits<E>::exponent_bias;
        auto exp = dpl::floating_point_traits<E>::exponent_mask;
        auto shift = dpl::countr_zero(exp);
        // Use quad root instead of square root
        auto qrt =
            dpl::bitset<dpl::type_bit_v<E>>(
                ((dpl::to_underlying(exp >> shift) - bias) >> 2) + bias - 1)
            << shift;
        return dpl::bit_cast<E>(qrt);
    }();
    template <typename E>
    static constexpr auto min = -max<E>;

    template <typename E>
    static constexpr auto mixmask() noexcept
    requires (fmop == dpp::fmaddsub || fmop == dpp::fmaddsac)
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
    requires (fmop == dpp::fmsubadd || fmop == dpp::fmsubacc)
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
    template <rng_like Rng, dpl::floating_point_like... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                if constexpr (dpp::is_simd_canonical_invocable<vec_t<E>,
                                  vec_t<E>, vec_t<E>>(fmop)) {
                    return fused_multiply::template run<E>(engine);
                } else {
                    return true;
                }
            },
            pack);
    }

    template <dpl::floating_point_like E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires (fmop == dpp::fmaddsub || fmop == dpp::fmsubadd ||
        fmop == dpp::fmaddsac || fmop == dpp::fmsubacc)
    {
        dpl::test::array_generator<A, E> const data_generator(min<E>, max<E>);
        dpl::test::scalar_generator<E> const src_generator(
            max<E> * max<E> * max<E>, dpp::max_value_v<E>);
        constexpr auto expected_op = [](vec_t<E> vlhs, vec_t<E> vmid,
                                         vec_t<E> vrhs) noexcept {
            if constexpr (fmop == dpp::fmaddsub || fmop == dpp::fmsubadd) {
                return dpp::select(mixmask<E>(), dpp::fmadd(vlhs, vmid, vrhs),
                    dpp::fmsub(vlhs, vmid, vrhs));
            } else {
                return dpp::select(mixmask<E>(), dpp::fmacc(vlhs, vmid, vrhs),
                    dpp::fmsac(vlhs, vmid, vrhs));
            }
        };
        for (auto i = 0zu; i < 4; ++i) {
            auto const lhs = data_generator(engine);
            auto const mid = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const src = src_generator(engine);
            auto const vlhs = dpp::load<E, A>(lhs.data());
            auto const vmid = dpp::load<E, A>(mid.data());
            auto const vrhs = dpp::load<E, A>(rhs.data());

            auto const vexpected = expected_op(vlhs, vmid, vrhs);
            auto const vactual = fmop(vlhs, vmid, vrhs);

            assert(dpp::all_of(vexpected == vactual));
            dpl::test::ternary_assignment<A>::template test_masked<E>(
                lhs, mid, rhs, fmop, src);

            if constexpr (dpp::is_simd_canonical_invocable<vec_t<E>, vec_t<E>,
                              vec_t<E>>(dpp::fmadd)) {
                if not consteval {
                    // catostrophic cancellation test
                    {
                        constexpr auto digitsm1 = dpl::countr_zero(
                            floating_point_traits<E>::exponent_mask);
                        // This is equivalent to 2^0 * 2^-digitsm1
                        constexpr auto small_exp =
                            floating_point_traits<E>::exponent_bias - digitsm1;
                        using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
                        constexpr auto small =
                            dpl::bit_cast<E>(bitset_t(small_exp) << digitsm1);
                        auto const vone = dpp::broadcast<A, E>(1);
                        auto const vsmall = dpp::broadcast<A, E>(small);
                        auto const a = dpp::add(vone, vsmall);
                        auto const b = dpp::subtract(vone, vsmall);
                        auto const c = dpp::negate(vone, mixmask<E>(), vone);
                        if constexpr (fmop == dpp::fmaddsub ||
                            fmop == dpp::fmsubadd) {
                            assert(dpp::all_of(
                                expected_op(a, b, c) == fmop(a, b, c)));
                        } else {
                            assert(dpp::all_of(
                                expected_op(c, a, b) == fmop(c, a, b)));
                        }
                    }
                    {
                        auto const vtwo = dpp::broadcast<A, E>(2);
                        auto const vlarge =
                            dpp::broadcast<A, E>(dpp::max_value);
                        auto const c =
                            dpp::negate(vlarge, mixmask<E>(), vlarge);
                        if constexpr (fmop == dpp::fmaddsub ||
                            fmop == dpp::fmsubadd) {
                            assert(dpp::all_of(expected_op(vlarge, vtwo, c) ==
                                fmop(vlarge, vtwo, c)));
                        } else {
                            assert(dpp::all_of(expected_op(c, vlarge, vtwo) ==
                                fmop(c, vlarge, vtwo)));
                        }
                    }
                }
            }
        }

        return true;
    }

    template <dpl::floating_point_like E, rng_like Rng>
    static constexpr bool run(Rng& engine) {

        dpl::test::array_generator<A, E> const data_generator(min<E>, max<E>);
        dpl::test::scalar_generator<E> const src_generator(
            max<E> * max<E> * max<E>, dpp::max_value_v<E>);

        auto const lhs = data_generator(engine);
        auto const mid = data_generator(engine);
        auto const rhs = data_generator(engine);
        auto const src = src_generator(engine);

        auto expected = lhs;
        for (auto i = 0zu; i < expected.size(); ++i) {
            expected[i] = expected_op<E>(lhs[i], mid[i], rhs[i]);
        }

        dpl::test::ternary_assignment<A>::template test<E>(
            lhs, mid, rhs, fmop, expected);
        dpl::test::ternary_assignment<A>::template test_masked<E>(
            lhs, mid, rhs, fmop, src);

        dpl::test::ternary_assignment<A>::template test<E>(
            1, 1, 1, fmop, expected_op<E>(1, 1, 1));
        if constexpr (fmop == dpp::fmadd || fmop == dpp::fmsub ||
            fmop == dpp::fnmadd || fmop == dpp::fnmsub) {
            dpl::test::ternary_assignment<A>::template test<E>(
                1, 1, 0, fmop, expected_op<E>(1, 1, 0));
            auto const a = src_generator(engine), b = src_generator(engine);
            dpl::test::ternary_assignment<A>::template test<E>(
                a, b, 0, fmop, expected_op<E>(a, b, 0));
        } else {
            dpl::test::ternary_assignment<A>::template test<E>(
                0, 1, 1, fmop, expected_op<E>(0, 1, 1));
            auto const a = src_generator(engine), b = src_generator(engine);
            dpl::test::ternary_assignment<A>::template test<E>(
                0, a, b, fmop, expected_op<E>(0, a, b));
        }

        auto const cancellation_tests = []() {
            auto const range =
                dpl::popcount(dpl::floating_point_traits<E>::exponent_mask);
            auto const drange = dpl::popcount(
                dpl::floating_point_traits<double>::exponent_mask);
            auto const precision = dpl::floating_point_traits<E>::digits;
            auto const dprecision = dpl::floating_point_traits<double>::digits;
            if consteval {
                return drange > drange && dprecision > precision;
            } else {
                return true;
            }
        }();

        if (cancellation_tests) {
            // catostrophic cancellation test
            constexpr auto digitsm1 =
                dpl::countr_zero(floating_point_traits<E>::exponent_mask);
            // This is equivalent to 2^0 * 2^-digitsm1
            constexpr auto small_exp =
                floating_point_traits<E>::exponent_bias - digitsm1;
            using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
            constexpr auto small =
                dpl::bit_cast<E>(bitset_t(small_exp) << digitsm1);

            if constexpr (fmop == dpp::fmadd) {
                dpl::test::ternary_assignment<A>::template test<E>(
                    1 + small, 1 - small, -1, fmop, -small * small);
                // overflow cancels
                dpl::test::ternary_assignment<A>::template test<E>(
                    dpp::max_value_v<E>, 2, -dpp::max_value_v<E>, fmop,
                    dpp::max_value_v<E>);
            } else if constexpr (fmop == dpp::fmsub) {
                dpl::test::ternary_assignment<A>::template test<E>(
                    1 + small, 1 - small, 1, fmop, -small * small);
                // overflow cancels
                dpl::test::ternary_assignment<A>::template test<E>(
                    dpp::max_value_v<E>, 2, dpp::max_value_v<E>, fmop,
                    dpp::max_value_v<E>);
            } else if constexpr (fmop == dpp::fnmadd) {
                dpl::test::ternary_assignment<A>::template test<E>(
                    1 + small, 1 - small, 1, fmop, small * small);
                // overflow cancels
                dpl::test::ternary_assignment<A>::template test<E>(
                    dpp::max_value_v<E>, 2, dpp::max_value_v<E>, fmop,
                    -dpp::max_value_v<E>);
            } else if constexpr (fmop == dpp::fnmsub) {
                dpl::test::ternary_assignment<A>::template test<E>(
                    1 + small, 1 - small, -1, fmop, small * small);
                // overflow cancels
                dpl::test::ternary_assignment<A>::template test<E>(
                    dpp::max_value_v<E>, 2, -dpp::max_value_v<E>, fmop,
                    -dpp::max_value_v<E>);
            }
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using fmadd = fused_multiply<dpp::fmadd, A>;
export template <dpp::simd_abi A>
using fmsub = fused_multiply<dpp::fmsub, A>;
export template <dpp::simd_abi A>
using fnmadd = fused_multiply<dpp::fnmadd, A>;
export template <dpp::simd_abi A>
using fnmsub = fused_multiply<dpp::fnmsub, A>;
export template <dpp::simd_abi A>
using fmaddsub = fused_multiply<dpp::fmaddsub, A>;
export template <dpp::simd_abi A>
using fmsubadd = fused_multiply<dpp::fmsubadd, A>;

export template <dpp::simd_abi A>
using fmacc = fused_multiply<dpp::fmacc, A>;
export template <dpp::simd_abi A>
using fmsac = fused_multiply<dpp::fmsac, A>;
export template <dpp::simd_abi A>
using fnmacc = fused_multiply<dpp::fnmacc, A>;
export template <dpp::simd_abi A>
using fnmsac = fused_multiply<dpp::fnmsac, A>;
export template <dpp::simd_abi A>
using fmaddsac = fused_multiply<dpp::fmaddsac, A>;
export template <dpp::simd_abi A>
using fmsubacc = fused_multiply<dpp::fmsubacc, A>;

} // namespace dpl::test
