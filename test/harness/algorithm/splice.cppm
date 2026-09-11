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
    using array_t = decltype(test::make_array<E, A>());
    template <typename E>
    using mask_t = dpp::make_canonical_mask_t<E, A>;

    template <dpp::simd_element_for<A> E>
    static constexpr array_t<E> expected_op(
        mask_t<E> mask, array_t<E> const& lhs, array_t<E> const& rhs) noexcept {
        auto const low = dpp::countr_zero(mask);
        if (low == abi_traits<E>::size()) {
            return rhs;
        }

        auto const high = dpp::countl_zero(mask);
        auto const split = abi_traits<E>::size() - high;
        array_t<E> result = lhs;
        for (auto const i : linear_counter(result)) {
            auto const idx = i + low;
            result[i] = idx < split ? lhs[idx] : rhs[idx - split];
        }

        return result;
    }

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto const lanes = abi_traits<E>::size();
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
    static constexpr bool run(Rng& engine) {
        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;
        for (auto const _ : test_count<E>()) {
            auto const mask = mask_generator(engine);
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const expected = expected_op<E>(mask, lhs, rhs);

            operation_fixture<A>::test(
                test::bitcmp, expected, dpp::splice, mask, lhs, rhs);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            run_const_mask_test<E>([&](auto cmask) {
                auto const lhs = data_generator(engine);
                auto const rhs = data_generator(engine);

                static_assert(dpp::is_simd_invocable<decltype(cmask),
                    dpp::make_canonical_vector_t<E, A>,
                    dpp::make_canonical_vector_t<E, A>>(dpp::splice));
                auto const expected = expected_op<E>(cmask, lhs, rhs);
                operation_fixture<A>::test(
                    test::bitcmp, expected, dpp::splice, cmask, lhs, rhs);
            });
        }

        return true;
    }
};

} // namespace dpl::test
