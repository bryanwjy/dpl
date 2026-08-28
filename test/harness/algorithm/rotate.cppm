// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.rotate;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_algorithm_operation auto rotateop, dpp::simd_abi A>
class lane_rotate {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return lane_rotate::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        dpl::test::array_generator<A, E> const data_generator;
        dpl::test::scalar_generator<size_t> const rotate_generator(
            0zu, abi_traits<E>::size());
        auto const lhs = data_generator(engine);
        auto expected = lhs;
        auto const src = -lhs[rotate_generator(engine)];

        for (auto x = 0; x < 3; ++x) {
            auto const rhs = rotate_generator(engine);
            for (auto i = 0zu; i < expected.size(); ++i) {
                if constexpr (rotateop == dpp::rotate_left) {
                    auto const j = i + rhs;
                    expected[i] =
                        j < expected.size() ? lhs[j] : lhs[j - expected.size()];
                } else {
                    auto const j = i - rhs;
                    expected[i] = j >= expected.size()
                        ? lhs[expected.size() + j]
                        : lhs[j];
                }
            }

            dpl::test::unary_transform<A>::template test<E>(
                lhs, [rhs](auto... args) { return rotateop(args..., rhs); },
                expected, test::bitcmp);
            dpl::test::unary_transform<A>::template test_masked<E>(
                lhs, [rhs](auto... args) { return rotateop(args..., rhs); },
                src);

            {
                // OOB test
                auto const oob = rhs + abi_traits<E>::size();
                dpl::test::unary_transform<A>::template test<E>(
                    lhs, [oob](auto... args) { return rotateop(args..., oob); },
                    expected, test::bitcmp);
            }
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto count = 4zu;
            constexpr auto rotates = []() {
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
                    auto const rotatei = [&rotates](auto... args) {
                        return rotateop(args..., rotates[I]);
                    };
                    for (auto i = 0zu; i < expected.size(); ++i) {
                        if constexpr (rotateop == dpp::rotate_left) {
                            auto const j = i + rotates[I];
                            expected[i] = j < expected.size()
                                ? lhs[j]
                                : lhs[j - expected.size()];
                        } else {
                            auto const j = i - rotates[I];
                            expected[i] = j >= expected.size()
                                ? lhs[expected.size() + j]
                                : lhs[j];
                        }
                    }

                    dpl::test::unary_transform<A>::template test<E>(
                        lhs, rotatei, expected, test::bitcmp);
                    dpl::test::unary_transform<A>::template test_masked<E>(
                        lhs, rotatei, src);
                },
                dpl::make_index_sequence<count>{});

            // OOB test
            expected = lhs;
            dpl::test::unary_transform<A>::template test<E>(
                lhs,
                [](auto... args) {
                    return rotateop(args..., abi_traits<E>::size);
                },
                expected, test::bitcmp);
        }

        return true;
    }
};

export template <dpp::simd_abi A>
using rotate_left = lane_rotate<dpp::rotate_left, A>;
export template <dpp::simd_abi A>
using rotate_right = lane_rotate<dpp::rotate_right, A>;

} // namespace dpl::test
