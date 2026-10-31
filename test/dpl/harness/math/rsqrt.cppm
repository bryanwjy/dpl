// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.math.rsqrt;
export import dpl.test.support;
export import mpfr.interface;
export import dpl.test.harness.math.unary;

import dpl;

export namespace testdata {
template <dpl::floating_point_like T>
inline constexpr mpfr::data_entry<T, 1zu> rsqrt[] = {};
template <dpl::floating_point_like T>
inline constexpr dataspan_t<T> entries<dpp::rsqrt, T> = dataspan_t<T>(rsqrt<T>);
} // namespace testdata

export namespace dpl::test {
namespace dpp = dpl::datapar;
template <dpp::simd_abi A>
class rsqrt_base {
protected:
    constexpr ~rsqrt_base() = default;

    template <typename E>
    static constexpr auto data = testdata::entries<dpp::rsqrt, E>;

    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = dpp::simd_abi_traits<A, E>::size();
        auto const limit = []() {
            if consteval {
                return 4zu;
            } else {
                return 128zu;
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

    template <typename E>
    struct generated_data {
        using array_type = decltype(make_array<E, A>());
        using residual_type = decltype(make_residual_array<E, A>());
        array_type input;
        array_type expected;
        residual_type residual;
    };

    template <typename E, rng_like Rng>
    static constexpr auto generate_data(Rng& rng) noexcept {
        auto const src = rsqrt_base::data<E>;
        auto input = make_array<E, A>();
        auto expected = make_array<E, A>();
        auto residual = make_residual_array<E, A>();

        using pair = array<decltype(make_array<E, A>()), 2>;
        for (auto const i : linear_counter(input)) {
            auto const entry = src[rng() % src.size()];
            input[i] = entry.input;
            expected[i] = entry.value;
            residual[i] = entry.residual;
        }

        return generated_data<E>{input, expected, residual};
    }
};

template <dpp::simd_abi A, double ulp>
class rsqrt_precise : rsqrt_base<A> {
    using rsqrt_base<A>::generated_data;
    using rsqrt_base<A>::run_const_mask_test;
    using rsqrt_base<A>::test_count;

public:
    template <rng_like Rng, dpl::floating_point_like... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return rsqrt_precise::template run<E>(engine);
            },
            pack);
    }

    template <dpl::floating_point_like E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);
        constexpr test::array_generator<A, E> src_generator;
        constexpr test::mask_generator<A, E> mask_generator;
        constexpr auto opt = dpl::accuracy::precise;
        for (auto const _ : rsqrt_precise::template test_count<E>()) {
            auto const [in, out, res] =
                rsqrt_precise::template generate_data<E>(engine);
            auto const src = src_generator(engine);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test(
                test::ulp_cmp<ulp>(test::span(res)), out, dpp::rsqrt, in, opt);
            operation_fixture<A>::test_masked(dpp::rsqrt, src, vtrue, in, opt);
            operation_fixture<A>::test_masked(dpp::rsqrt, src, vfalse, in, opt);
            operation_fixture<A>::test_masked(dpp::rsqrt, src, mask, in, opt);
            operation_fixture<A>::test_masked(
                dpp::rsqrt, dpp::zero, mask, in, opt);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            [[maybe_unused]] auto const [in, out, res] =
                rsqrt_precise::template generate_data<E>(engine);
            auto const src = src_generator(engine);
            rsqrt_precise::template run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(
                    dpp::rsqrt, src, cmask, in, opt);
                operation_fixture<A>::test_masked(
                    dpp::rsqrt, dpp::zero, cmask, in, opt);
            });
        }

        return true;
    }
};

template <dpp::simd_abi A>
class rsqrt_approx : rsqrt_base<A> {
    using rsqrt_base<A>::generated_data;
    using rsqrt_base<A>::run_const_mask_test;
    using rsqrt_base<A>::test_count;

public:
    template <rng_like Rng, dpl::floating_point_like... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return rsqrt_approx::template run<E>(engine);
            },
            pack);
    }

    template <dpl::floating_point_like E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        auto const vtrue = dpp::broadcast<E, A>(true);
        auto const vfalse = dpp::broadcast<E, A>(false);
        constexpr test::array_generator<A, E> src_generator;
        constexpr test::mask_generator<A, E> mask_generator;
        constexpr auto opt = dpl::accuracy::approximate;
        constexpr auto p = dpl::floating_point_traits<E>::digits / 2;
        for (auto const _ : rsqrt_approx::template test_count<E>()) {
            auto const [in, out, res] =
                rsqrt_approx::template generate_data<E>(engine);
            auto const src = src_generator(engine);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test(
                test::precision_cmp<p>, out, dpp::rsqrt, in, opt);
            operation_fixture<A>::test_masked(dpp::rsqrt, src, vtrue, in, opt);
            operation_fixture<A>::test_masked(dpp::rsqrt, src, vfalse, in, opt);
            operation_fixture<A>::test_masked(dpp::rsqrt, src, mask, in, opt);
            operation_fixture<A>::test_masked(
                dpp::rsqrt, dpp::zero, mask, in, opt);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            [[maybe_unused]] auto const [in, out, res] =
                rsqrt_approx::template generate_data<E>(engine);
            auto const src = src_generator(engine);
            rsqrt_approx::template run_const_mask_test<E>([&](auto cmask) {
                operation_fixture<A>::test_masked(
                    dpp::rsqrt, src, cmask, in, opt);
                operation_fixture<A>::test_masked(
                    dpp::rsqrt, dpp::zero, cmask, in, opt);
            });
        }

        return true;
    }
};
} // namespace dpl::test
