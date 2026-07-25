// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.permutation;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class permutation {
public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return permutation::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<A, E> const data_generator(test::half_range);
        using index_t = dpp::signed_representation_t<E>;
        constexpr auto lanes = dpp::simd_abi_traits<A, E>::size();
        test::array_generator<A, index_t> const idx_generator(0, lanes);
        test::array_generator<A, index_t> const oob_generator(0, 2 * lanes);
        test::scalar_generator<E> const src_generator(
            dpp::max_value_v<E> / 4 * 3, dpp::max_value_v<E>);

        for (auto i = 0; i < 3; ++i) {
            auto const lhs = data_generator(engine);
            auto const rhs = idx_generator(engine);
            auto const oob = oob_generator(engine);
            auto const src = src_generator(engine);

            auto expected = lhs;
            {
                for (auto i = 0zu; i < expected.size(); ++i) {
                    expected[i] = lhs[rhs[i]];
                }
                auto const vlhs = dpp::load<A>(lhs.data());
                auto const vrhs = dpp::load<A>(rhs.data());
                auto const vexpected = dpp::load<A>(expected.data());
                auto const vactual = dpp::permute(vlhs, vrhs);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                test::unary_transform<A>::template test_masked<E>(
                    lhs,
                    [&vrhs](
                        auto... args) { return dpp::permute(args..., vrhs); },
                    src);
            }

            {
                for (auto i = 0zu; i < expected.size(); ++i) {
                    if (oob[i] < lhs.size()) {
                        expected[i] = lhs[oob[i]];
                    } else {
                        expected[i] = 0;
                    }
                }

                auto const vlhs = dpp::load<A>(lhs.data());
                auto const vrhs = dpp::load<A>(oob.data());
                auto const vexpected = dpp::load<A>(expected.data());
                auto const vactual = dpp::permute(vlhs, vrhs);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
            }
        }

        {
            auto const lhs = data_generator(engine);
            auto const vrhs = dpp::lane_index<E, A>();
            auto const vlhs = dpp::load<A>(lhs.data());
            auto const vactual = dpp::permute(vlhs, vrhs);
            assert(dpp::all_of(test::bitcmp(vactual, vlhs)));
        }

        {
            auto const lhs = data_generator(engine);
            auto const idx = test::scalar_generator<index_t>(0, lanes)(engine);
            auto const vrhs = dpp::broadcast<A>(idx);
            auto const vlhs = dpp::load<A>(lhs.data());
            auto const vactual = dpp::permute(vlhs, vrhs);
            auto const vexpected = dpp::broadcast<E, A>(lhs[idx]);
            assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto iota = dpl::make_index_sequence<lanes>{};
            constexpr auto riota = [lanes]<size_t... Is>(
                                       index_sequence<Is...>) {
                return index_sequence<(lanes - 1 - Is)...>{};
            }(iota);
            constexpr auto constant = [lanes]<size_t... Is>(
                                          index_sequence<Is...>) {
                constexpr auto mid = lanes / 2;
                return index_sequence<((Is == Is) ? mid : mid)...>{};
            }(iota);
            constexpr auto rot1 = [lanes]<size_t... Is>(index_sequence<Is...>) {
                return index_sequence<((Is + 1) % lanes)...>{};
            }(iota);
            constexpr auto inter = []<size_t... Is>(index_sequence<Is...>) {
                return index_sequence<(Is * 2)..., (Is * 2 + 1)...>{};
            }(dpl::make_index_sequence<lanes / 2>{});
            auto const expected_op = [iota]<size_t... Is>(
                                         auto lhs, index_sequence<Is...>) {
                auto result = lhs;
                dpl::apply(
                    [&](auto... idx) { (..., (result[idx] = lhs[Is])); }, iota);
                return result;
            };

            {
                auto const lhs = data_generator(engine);
                auto const vlhs = dpp::load<A>(lhs.data());
                auto const vactual = dpp::permute(vlhs, iota);
                auto const vexpected = vlhs;
                auto const src = src_generator(engine);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                test::unary_transform<A>::template test_masked<E>(
                    lhs,
                    [iota](
                        auto... args) { return dpp::permute(args..., iota); },
                    src);
            }

            {
                auto const lhs = data_generator(engine);
                auto const vlhs = dpp::load<A>(lhs.data());
                auto const vactual = dpp::permute(vlhs, riota);
                auto const vexpected =
                    dpp::load<A>(expected_op(lhs, riota).data());
                auto const src = src_generator(engine);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                test::unary_transform<A>::template test_masked<E>(
                    lhs,
                    [riota](
                        auto... args) { return dpp::permute(args..., riota); },
                    src);
            }

            {
                auto const lhs = data_generator(engine);
                auto const vlhs = dpp::load<A>(lhs.data());
                auto const vactual = dpp::permute(vlhs, constant);
                auto const vexpected =
                    dpp::load<A>(expected_op(lhs, constant).data());
                auto const src = src_generator(engine);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                test::unary_transform<A>::template test_masked<E>(
                    lhs,
                    [constant](auto... args) {
                        return dpp::permute(args..., constant);
                    },
                    src);
            }

            {
                auto const lhs = data_generator(engine);
                auto const vlhs = dpp::load<A>(lhs.data());
                auto const vactual = dpp::permute(vlhs, rot1);
                auto const vexpected =
                    dpp::load<A>(expected_op(lhs, rot1).data());
                auto const src = src_generator(engine);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                test::unary_transform<A>::template test_masked<E>(
                    lhs,
                    [rot1](
                        auto... args) { return dpp::permute(args..., rot1); },
                    src);
            }

            {
                auto const lhs = data_generator(engine);
                auto const vlhs = dpp::load<A>(lhs.data());
                auto const vactual = dpp::permute(vlhs, inter);
                auto const vexpected =
                    dpp::load<A>(expected_op(lhs, inter).data());
                auto const src = src_generator(engine);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                test::unary_transform<A>::template test_masked<E>(
                    lhs,
                    [inter](
                        auto... args) { return dpp::permute(args..., inter); },
                    src);
            }
        }

        return true;
    }
};
} // namespace dpl::test
