// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.lane_broadcast;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class lane_broadcast {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return lane_broadcast::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<A, E> const data_generator(test::half_range);
        test::scalar_generator<E> const src_generator(
            dpp::max_value_v<E> / 4 * 3, dpp::max_value_v<E>);
        auto const lhs = data_generator(engine);
        auto const src = src_generator(engine);
        constexpr auto count = dpp::min(abi_traits<E>::size(), 4zu);
        auto expected = lhs;

        for (auto i = 0zu; i < count; ++i) {
            for (auto& val : expected) {
                val = lhs[i];
            }

            auto const broadcast = [i](auto... args) {
                return dpp::broadcast_lane(args..., i);
            };

            dpl::test::unary_transform<A>::template test<E>(
                lhs, broadcast, expected, test::bitcmp);
            dpl::test::unary_transform<A>::template test_masked<E>(
                lhs, broadcast, src);
        }
        {
            // OOB results in zero
            for (auto& val : expected) {
                val = 0;
            }
            dpl::test::unary_transform<A>::template test<E>(
                lhs,
                [](auto lhs) {
                    return dpp::broadcast_lane(lhs, abi_traits<E>::size());
                },
                expected);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            dpl::pack::for_each(
                [&]<size_t I>(dpl::size_constant<I>) {
                    constexpr auto offset = [count]() {
                        if constexpr (abi_traits<E>::size() > count) {
                            test::mt19937 rng{};
                            return test::scalar_generator<size_t>(0zu, count)(
                                rng);
                        } else {
                            return 0zu;
                        }
                    }();
                    constexpr auto stride =
                        abi_traits<E>::size() > count ? count : 1zu;
                    constexpr auto idx = dpp::min(
                        I * stride + offset, abi_traits<E>::size() - 1);
                    auto const broadcasti = [idx](auto... args) {
                        return dpp::broadcast_lane(args..., dpp::imm<idx>);
                    };

                    for (auto& val : expected) {
                        val = lhs[idx];
                    }

                    dpl::test::unary_transform<A>::template test<E>(
                        lhs, broadcasti, expected, test::bitcmp);
                    dpl::test::unary_transform<A>::template test_masked<E>(
                        lhs, broadcasti, src);
                },
                dpl::make_index_sequence<count>{});
        }

        return true;
    }
};

} // namespace dpl::test
