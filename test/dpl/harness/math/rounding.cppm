// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.math.rounding;
export import dpl.test.support;
export import mpfr.interface;

import dpl;

export namespace testdata {
using namespace dpl::ext::literals; // NOLINT
// The following are meant to be specialized by
// the test drivers using the mpfr generator
template <dpl::floating_point_like T>
inline constexpr mpfr::data_entry<T, 1zu> ceil[] = {};
template <dpl::floating_point_like T>
inline constexpr mpfr::data_entry<T, 1zu> floor[] = {};
template <dpl::floating_point_like T>
inline constexpr mpfr::data_entry<T, 1zu> roundeven[] = {};
template <dpl::floating_point_like T>
inline constexpr mpfr::data_entry<T, 1zu> trunc[] = {};
template <dpl::floating_point_like T>
inline constexpr mpfr::data_entry<T, 1zu> round[] = {};
} // namespace testdata

export namespace dpl::test {
namespace dpp = dpl::datapar;
template <dpp::simd_math_operation auto rop, dpp::simd_abi A>
class rounding {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = dpp::simd_abi_traits<A, E>::size();
        auto const limit = []() {
            if consteval {
                return 4zu;
            } else {
                return 32zu;
            }
        }();

        auto max = lanes >= dpl::type_bit_v<size_t>
            ? limit
            : static_cast<size_t>((1zu << lanes) - 1);
        return linear_counter(max < limit ? max : limit);
    }

    template <typename E>
    static constexpr void run_const_mask_test(auto func) noexcept {
        [&]<size_t I = 0, serialized_mt19937 S = {}>(this auto self,
            dpp::immediate<I> = dpp::imm<I>, mt19937_type<S> = {}) {
            if constexpr (I < test_count<E>().size()) {
                constexpr auto lanes = dpp::simd_abi_traits<A, E>::size();
                constexpr bit_generator<lanes> bitgen;
                constexpr auto pair = S.generate_with(bitgen);
                constexpr auto cmask = dpp::deduce_const_mask_v<pair.value>;

                func(cmask);
                self(dpp::imm<I + 1>, mt19937_type<pair.state>{});
            }
        }
        ();
    }

    template <typename E, rng_like Rng>
    static constexpr auto generate_data(Rng& rng) noexcept {
        auto const src = [&] {
            if constexpr (rop == dpp::ceil) {
                return test::span{testdata::ceil<E>};
            } else if constexpr (rop == dpp::floor) {
                return test::span{testdata::floor<E>};
            } else if constexpr (rop == dpp::trunc) {
                return test::span{testdata::trunc<E>};
            } else if constexpr (rop == dpp::roundeven) {
                return test::span{testdata::roundeven<E>};
            } else {
                return test::span{testdata::round<E>};
            }
        }();

        auto input = make_array<E, A>();
        auto expected = make_array<E, A>();
        using pair = array<decltype(make_array<E, A>()), 2>;
        for (auto const i : linear_counter(input)) {
            auto const entry = src[rng() % src.size()];
            input[i] = entry.input;
            expected[i] = entry.value;
        }

        return pair{input, expected};
    }

    static consteval auto rounding_flag() noexcept
    requires (rop != dpp::round)
    {
        if constexpr (rop == dpp::ceil) {
            return dpp::rounding::to_pos_inf;
        } else if constexpr (rop == dpp::floor) {
            return dpp::rounding::to_neg_inf;
        } else if constexpr (rop == dpp::trunc) {
            return dpp::rounding::to_zero;
        } else {
            static_assert(rop == dpp::roundeven);
            return dpp::rounding::to_nearest_int;
        }
    }

public:
    template <rng_like Rng, dpl::floating_point_like... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return rounding::template run<E>(engine);
            },
            pack);
    }

    template <dpl::floating_point_like E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);
        constexpr test::array_generator<A, E> src_generator;
        constexpr test::mask_generator<A, E> mask_generator;
        for (auto const _ : test_count<E>()) {
            auto const [in, out] = generate_data<E>(engine);
            auto const src = src_generator(engine);
            auto const mask = mask_generator(engine);
            test::operation_fixture<A>::test(
                out, rop, in, dpp::rounding::no_exc);
            operation_fixture<A>::test_masked(
                rop, src, vtrue, in, dpp::rounding::no_exc);
            operation_fixture<A>::test_masked(
                rop, src, vfalse, in, dpp::rounding::no_exc);
            operation_fixture<A>::test_masked(
                rop, src, mask, in, dpp::rounding::no_exc);
            operation_fixture<A>::test_masked(
                rop, dpp::zero, mask, in, dpp::rounding::no_exc);
            if not consteval {
                test::operation_fixture<A>::test(out, rop, in);
                operation_fixture<A>::test_masked(rop, src, vtrue, in);
                operation_fixture<A>::test_masked(rop, src, vfalse, in);
                operation_fixture<A>::test_masked(rop, src, mask, in);
                operation_fixture<A>::test_masked(rop, dpp::zero, mask, in);
            }

            if constexpr (requires { rounding_flag(); }) {
                constexpr auto flag = rounding_flag() | dpp::rounding::no_exc;
                auto const vin = dpp::load<A>(in.data());
                // round can alias all other rounding methods by explicitly
                // specifying rounding flags
                auto const alias = dpp::round(vin, flag);
                assert(dpp::all_of(
                    bitcmp(alias, rop(vin, dpp::rounding::no_exc))));
            }
        }

        operation_fixture<A>::test(test::splat<E>(dpp::infinity), rop,
            test::splat<E>(dpp::infinity), dpp::rounding::no_exc);
        operation_fixture<A>::test(test::splat<E>(-dpp::infinity), rop,
            test::splat<E>(-dpp::infinity), dpp::rounding::no_exc);
        operation_fixture<A>::test(test::splat<E>(-dpp::infinity), rop,
            test::splat<E>(-dpp::infinity), dpp::rounding::no_exc);
        operation_fixture<A>::test(test::nancmp, test::splat<E>(dpp::nan), rop,
            test::splat<E>(dpp::nan), dpp::rounding::no_exc);

        return true;
    }
};

template <dpp::simd_abi A>
using ceil = rounding<dpp::ceil, A>;
template <dpp::simd_abi A>
using floor = rounding<dpp::floor, A>;
template <dpp::simd_abi A>
using roundeven = rounding<dpp::roundeven, A>;
template <dpp::simd_abi A>
using trunc = rounding<dpp::trunc, A>;
template <dpp::simd_abi A>
using round = rounding<dpp::round, A>;

} // namespace dpl::test
