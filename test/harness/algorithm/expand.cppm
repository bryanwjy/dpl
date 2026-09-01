// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.expand;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class expand {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using data = test::array<E, abi_traits<E>::size>;

    template <dpp::simd_element_for<A> E>
    static constexpr auto expected_op(data<E> const& val,
        dpl::bitset<abi_traits<E>::size> mask, data<E> const& src) noexcept
    requires dpp::fixed_width_abi<A>
    {
        assert(val.size() == src.size());
        data<E> result = val;
        for (auto i = 0zu, j = 0zu; i < val.size(); ++i) {
            if (mask[i]) {
                result[i] = val[j++];
            } else {
                result[i] = src[i];
            }
        }

        return result;
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return expand::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        constexpr auto width = abi_traits<E>::size();
        test::array_generator<A, E> const data_generator(test::half_range);
        test::array_generator<A, E> const src_generator(
            dpp::max_value_v<E> / 4 * 3, dpp::max_value_v<E>);
        test::bit_generator<width> mask_generator;
        auto const loop_count = []() {
            if consteval {
                return 3;
            } else {
                return 128;
            }
        }();

        for (auto i = 0; i < loop_count; ++i) {
            auto const val = data_generator(engine);
            auto const mask = mask_generator(engine);
            auto const src = src_generator(engine);
            auto const vval = dpp::load<A>(val.data());
            auto const vmask = dpp::from_bitset<A, E>(mask);

            {
                auto const expected = expected_op(val, mask, src);
                auto const vexpected = dpp::load<A>(expected.data());
                auto const vsrc = dpp::load<A>(src.data());
                auto const vactual = dpp::expand(vval, vmask, vsrc);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
            }
            {
                auto const expected = expected_op(val, mask, decltype(val)());
                auto const vexpected = dpp::load<A>(expected.data());
                auto const vactual = dpp::expand(vval, vmask, dpp::zero);
                assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
            }
        }

        {
            constexpr auto count = 3zu;
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
                    auto const val = data_generator(engine);
                    auto const src = src_generator(engine);
                    auto const cmask = dpp::const_mask<masks[I].size(),
                        dpl::to_underlying(masks[I])>{};

                    auto const vval = dpp::load<A>(val.data());
                    {
                        auto const expected = expected_op(val, masks[I], src);
                        auto const vexpected = dpp::load<A>(expected.data());
                        auto const vsrc = dpp::load<A>(src.data());
                        auto const vactual = dpp::expand(vval, cmask, vsrc);
                        assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                    }

                    {
                        auto const expected =
                            expected_op(val, masks[I], decltype(val)());
                        auto const vexpected = dpp::load<A>(expected.data());
                        auto const vactual =
                            dpp::expand(vval, cmask, dpp::zero);
                        assert(dpp::all_of(test::bitcmp(vactual, vexpected)));
                    }
                },
                dpl::make_index_sequence<count>{});
        }

        return true;
    }
};
} // namespace dpl::test
