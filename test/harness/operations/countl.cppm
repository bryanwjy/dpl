// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.countl;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class countl {
    using abi_t = A;
    template <typename E>
    using vec_t = dpp::basic_vector<E, A>;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return countl::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        // Go through a few iterations
        using count_t = dpp::unsigned_representation_t<E>;
        test::array_generator<A, E> const data_generator;
        test::array_generator<A, count_t> const src_generator(
            dpl::type_bit_v<E>, dpl::integral_traits<count_t>::max_value);
        for (auto i = 0zu; i < 3; ++i) {
            auto const data = data_generator(engine);
            auto const src = src_generator(engine);
            auto const expected = [&]() {
                auto result = src;
                for (auto i = 0zu; i < result.size(); ++i) {
                    result[i] = dpl::countl_zero(dpl::to_unsigned(data[i]));
                }
                return result;
            }();

            test::unary_transform<A>::template test<E>(
                data, dpp::countl_zero, expected);
            auto const idx = i < src.size() ? i : i % src.size();
            test::unary_transform<A>::template test_masked<E>(
                data, dpp::countl_zero, src[idx]);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            test::bit_generator<abi_traits<E>::size> const bit_generator;
            for (auto i = 0zu; i < 3; ++i) {
                auto const data = bit_generator(engine);
                auto const expected = dpl::to_unsigned(dpl::countl_zero(data));
                assert(
                    dpp::countl_zero(dpp::from_bitset<A, E>(data)) == expected);
            }
        }

        for (auto i = 0zu; i < 3; ++i) {
            auto const data = data_generator(engine);
            auto const src = src_generator(engine);
            auto const expected = [&]() {
                auto result = src;
                for (auto i = 0zu; i < result.size(); ++i) {
                    result[i] = dpl::countl_one(dpl::to_unsigned(data[i]));
                }
                return result;
            }();

            test::unary_transform<A>::template test<E>(
                data, dpp::countl_one, expected);
            auto const idx = i < src.size() ? i : i % src.size();
            test::unary_transform<A>::template test_masked<E>(
                data, dpp::countl_one, src[idx]);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            test::bit_generator<abi_traits<E>::size> const bit_generator;
            for (auto i = 0zu; i < 3; ++i) {
                auto const data = bit_generator(engine);
                auto const expected = dpl::to_unsigned(dpl::countl_one(data));
                assert(
                    dpp::countl_one(dpp::from_bitset<A, E>(data)) == expected);
            }
        }

        return true;
    }
};

} // namespace dpl::test
