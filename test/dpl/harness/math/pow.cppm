// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.math.pow;
export import dpl.test.support;
export import mpfr.interface;

import dpl;

export namespace testdata {
using namespace dpl::ext::literals; // NOLINT
namespace dpp = dpl::datapar;
template <dpl::floating_point_like E>
inline constexpr mpfr::data_entry<E, 2zu> pow[] = {};
} // namespace testdata

export namespace dpl::test {
namespace dpp = dpl::datapar;
template <dpp::simd_abi A, double ulp>
class pow {

    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = dpp::simd_abi_traits<A, E>::size();
        auto const limit = []() {
            if consteval {
                return 4zu;
            } else {
                return 256zu;
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
        array_type left;
        array_type right;
        array_type expected;
        residual_type residual;
    };

    template <typename E, rng_like Rng>
    static constexpr auto generate_data(Rng& rng) noexcept {
        auto const src = test::span{testdata::pow<E>};
        auto left = make_array<E, A>();
        auto right = make_array<E, A>();
        auto expected = make_array<E, A>();
        auto residual = make_residual_array<E, A>();

        for (auto const i : linear_counter(left)) {
            auto const entry = src[rng() % src.size()];
            left[i] = entry.input[0];
            right[i] = entry.input[1];
            expected[i] = entry.value;
            residual[i] = entry.residual;
        }

        return generated_data<E>{left, right, expected, residual};
    }

public:
    template <rng_like Rng, dpl::floating_point_like... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return pow::template run<E>(engine);
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
            auto const [lin, rin, out, res] = generate_data<E>(engine);
            auto const src = src_generator(engine);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test(
                test::ulp_cmp<ulp>(test::span(res)), out, dpp::pow, lin, rin);
            operation_fixture<A>::test_masked(dpp::pow, src, vtrue, lin, rin);
            operation_fixture<A>::test_masked(dpp::pow, src, vfalse, lin, rin);
            operation_fixture<A>::test_masked(dpp::pow, src, mask, lin, rin);
            operation_fixture<A>::test_masked(
                dpp::pow, dpp::zero, mask, lin, rin);
        }

        return true;
    }
};
} // namespace dpl::test
