// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.shift;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_algorithm_operation auto shiftop, dpp::simd_abi A>
class lane_shift {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return lane_shift::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        dpl::test::array_generator<A, E> const data_generator;
        dpl::test::scalar_generator<size_t> const shift_generator(
            0zu, abi_traits<E>::size());
        auto const lhs = data_generator(engine);
        auto expected = lhs;
        auto const src = -lhs[shift_generator(engine)];

        for (auto x = 0; x < 3; ++x) {
            auto const rhs = shift_generator(engine);
            for (auto i = 0zu; i < expected.size(); ++i) {
                if constexpr (shiftop == dpp::shift_left) {
                    auto const j = i + rhs;
                    expected[i] =
                        j < expected.size() ? lhs[j] : static_cast<E>(0);
                } else {
                    auto const j = i - rhs;
                    expected[i] =
                        j < expected.size() ? lhs[j] : static_cast<E>(0);
                }
            }

            dpl::test::unary_transform<A>::template test<E>(
                lhs, [rhs](auto... args) { return shiftop(args..., rhs); },
                expected, test::bitcmp);
            dpl::test::unary_transform<A>::template test_masked<E>(
                lhs, [rhs](auto... args) { return shiftop(args..., rhs); },
                src);

            {
                // OOB test
                auto const oob =
                    shift_generator(engine) + abi_traits<E>::size();
                dpl::test::unary_transform<A>::template test<E>(
                    lhs, [oob](auto... args) { return shiftop(args..., oob); },
                    decltype(lhs){}, test::bitcmp);
            }
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto count = 4zu;
            constexpr auto shifts = []() {
                return dpl::apply(
                    [](auto... idx) {
                        test::mt19937 rng{};
                        return array<size_t, count>{
                            (dpl::test::scalar_generator<size_t>(
                                idx * 0, abi_traits<E>::size())(rng))...};
                    },
                    dpl::make_index_sequence<count>{});
            }();
            dpl::pack::for_each(
                [&]<size_t I>(dpl::size_constant<I>) {
                    auto const shifti = [&shifts](auto... args) {
                        return shiftop(args..., shifts[I]);
                    };
                    for (auto i = 0zu; i < expected.size(); ++i) {
                        if constexpr (shiftop == dpp::shift_left) {
                            auto const j = i + shifts[I];
                            expected[i] = j < expected.size()
                                ? lhs[j]
                                : static_cast<E>(0);
                        } else {
                            auto const j = i - shifts[I];
                            expected[i] = j < expected.size()
                                ? lhs[j]
                                : static_cast<E>(0);
                        }
                    }

                    dpl::test::unary_transform<A>::template test<E>(
                        lhs, shifti, expected, test::bitcmp);
                    dpl::test::unary_transform<A>::template test_masked<E>(
                        lhs, shifti, src);
                },
                dpl::make_index_sequence<count>{});

            // OOB test
            dpl::test::unary_transform<A>::template test<E>(
                lhs,
                [](auto... args) {
                    return shiftop(args..., abi_traits<E>::size);
                },
                decltype(lhs){}, test::bitcmp);
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using shift_left = lane_shift<dpp::shift_left, A>;
export template <dpp::simd_abi A>
using shift_right = lane_shift<dpp::shift_right, A>;

} // namespace dpl::test