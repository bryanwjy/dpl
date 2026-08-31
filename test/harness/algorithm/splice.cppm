// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.splice;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class splice {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using vector = dpp::basic_vector<E, A>;
    template <typename E>
    using data = test::array<E, abi_traits<E>::size>;

    template <dpp::simd_element_for<A> E>
    static constexpr auto expected_op(dpl::bitset<abi_traits<E>::size> mask,
        data<E> const& lhs, data<E> const& rhs) noexcept
    requires dpp::fixed_width_abi<A>
    {
        auto const low = dpl::countr_zero(mask);
        if (low == abi_traits<E>::size()) {
            return rhs;
        }

        auto const high = dpl::countl_zero(mask);
        auto const split = abi_traits<E>::size() - high;
        data<E> result = lhs;
        for (auto i = 0zu; i < abi_traits<E>::size(); ++i) {
            auto const idx = i + low;
            result[i] = idx < split ? lhs[idx] : rhs[idx - split];
        }

        return result;
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return splice::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        dpl::test::array_generator<A, E> const data_generator;
        dpl::test::scalar_generator<size_t> const shift_generator(
            0zu, abi_traits<E>::size());
        dpl::test::bit_generator<abi_traits<E>::size()> const mask_generator;
        for (auto x = 0; x < 3; ++x) {
            auto const mask = mask_generator(engine);
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const expected = expected_op(mask, lhs, rhs);

            dpl::test::binary_transform<A>::template test<E>(
                lhs, rhs,
                [&mask](auto lhs, auto rhs) {
                    return dpp::splice(dpp::from_bitset<E, A>(mask), lhs, rhs);
                },
                expected, test::bitcmp);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            constexpr auto count = 4zu;
            constexpr auto masks = []() {
                return dpl::apply(
                    [](auto... idx) {
                        test::mt19937 rng{};
                        return array<dpl::bitset<abi_traits<E>::size()>, count>{
                            (dpl::test::bit_generator<abi_traits<E>::size() +
                                idx * 0>()(rng))...};
                    },
                    dpl::make_index_sequence<count>{});
            }();

            dpl::pack::for_each(
                [&]<size_t I>(dpl::size_constant<I>) {
                    auto const splicei = [&masks](auto... args) {
                        using cmask_t = dpp::const_mask<masks[I].size(),
                            dpl::to_underlying(masks[I])>;
                        return dpp::splice(cmask_t{}, args...);
                    };

                    auto const lhs = data_generator(engine);
                    auto const rhs = data_generator(engine);
                    auto const expected = expected_op(masks[I], lhs, rhs);

                    dpl::test::binary_transform<A>::template test<E>(
                        lhs, rhs, splicei, expected, test::bitcmp);
                },
                dpl::make_index_sequence<count>{});
        }

        return true;
    }
};

} // namespace dpl::test
