// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.byteswap;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class byteswap {
public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return byteswap::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        using bitset_t = dpl::bitset<dpl::type_bit_v<E>>;
        constexpr auto zeroed_bits = dpl::type_bit_v<E> / dpl::char_bit_v;
        constexpr auto min = [zeroed_bits]() {
            if constexpr (dpl::signed_integral<E>) {
                return dpl::bit_cast<E>(~bitset_t() << zeroed_bits);
            } else {
                return static_cast<E>(0);
            }
        }();
        constexpr auto max = [min, zeroed_bits]() {
            if constexpr (dpl::signed_integral<E>) {
                return dpl::bit_cast<E>(dpl::bit_cast<bitset_t>(min) ^
                    (~bitset_t() << (dpl::type_bit_v<E> - 1)));
            } else {
                return dpl::bit_cast<E>(~bitset_t() << zeroed_bits);
            }
        }();

        test::array_generator<A, E> const data_generator(min, max);
        test::scalar_generator<E> const src_generator(
            (-1 << zeroed_bits) + 1, 1 << zeroed_bits);

        for (auto i = 0zu; i < 4; ++i) {
            auto const data = data_generator(engine);
            auto const src = src_generator(engine);
            auto const expected = [&]() {
                auto result = data;
                for (auto i = 0zu; i < result.size(); ++i) {
                    result[i] = dpl::byteswap(data[i]);
                }
                return result;
            }();

            test::unary_transform<A>::template test<E>(
                data, dpp::byteswap, expected);
            test::unary_transform<A>::template test_masked<E>(
                data, dpp::byteswap, src);
        }

        return true;
    }
};

} // namespace dpl::test
