// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.operations.selection;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class selection {
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr linear_counter test_count() noexcept {
        auto lanes = dpp::simd_abi_traits<A, E>::size();
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

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr void vector_run(Rng& engine) {
        test::array_generator<A, E> const data_generator;
        test::mask_generator<A, E> const mask_generator;
        auto const vzero = dpp::broadcast<A, E>(dpp::zero);

        for (auto const _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const mmask = mask_generator(engine);
            constexpr bool negatable = requires {
                { !mmask } -> dpp::simd_mask;
            };
            auto const amask = test::to_mask_array<E, A>(mmask, vzero);
            auto expected = [&]() {
                auto result = test::make_array<E, A>();
                for (auto const i : linear_counter(amask)) {
                    result[i] = amask[i] != 0 ? lhs[i] : rhs[i];
                }
                return result;
            }();

            test::operation_fixture<A>::test(
                test::bitcmp, expected, dpp::select, mmask, lhs, rhs);
            if constexpr (negatable) {
                test::operation_fixture<A>::test(
                    test::bitcmp, expected, dpp::select, !mmask, rhs, lhs);
            }

            for (auto const i : linear_counter(amask)) {
                if (amask[i] == 0) {
                    expected[i] = 0;
                }
            }

            test::operation_fixture<A>::test(
                test::bitcmp, expected, dpp::select, mmask, lhs, dpp::zero);
            if constexpr (negatable)
                test::operation_fixture<A>::test(test::bitcmp, expected,
                    dpp::select, !mmask, dpp::zero, lhs);

            for (auto const i : linear_counter(amask)) {
                if (amask[i] != 0) {
                    expected[i] = 0;
                } else {
                    expected[i] = lhs[i];
                }
            }

            test::operation_fixture<A>::test(
                test::bitcmp, expected, dpp::select, mmask, dpp::zero, lhs);
            if constexpr (negatable)
                test::operation_fixture<A>::test(test::bitcmp, expected,
                    dpp::select, !mmask, lhs, dpp::zero);

            for (auto const i : linear_counter(amask)) {
                if (amask[i] == 0) {
                    expected[i] =
                        dpl::bit_cast<E>(~dpl::bit_representation_t<E>());
                } else {
                    expected[i] = lhs[i];
                }
            }

            test::operation_fixture<A>::test(
                test::bitcmp, expected, dpp::select, mmask, lhs, dpp::all_bits);
            if constexpr (negatable)
                test::operation_fixture<A>::test(test::bitcmp, expected,
                    dpp::select, !mmask, dpp::all_bits, lhs);

            for (auto const i : linear_counter(amask)) {
                if (amask[i] != 0) {
                    expected[i] =
                        dpl::bit_cast<E>(~dpl::bit_representation_t<E>());
                } else {
                    expected[i] = lhs[i];
                }
            }

            test::operation_fixture<A>::test(
                test::bitcmp, expected, dpp::select, mmask, dpp::all_bits, lhs);
            if constexpr (negatable)
                test::operation_fixture<A>::test(test::bitcmp, expected,
                    dpp::select, !mmask, lhs, dpp::all_bits);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const vlhs = dpp::load<A>(data_generator(engine).data());
            auto const vrhs = dpp::load<A>(data_generator(engine).data());
            run_const_mask_test<E>([&](auto cmask) {
                auto const mmask =
                    dpp::from_bitset<A, E>(dpp::to_bitset(cmask));
                test::operation_fixture<A>::test(test::bitcmp,
                    dpp::select(mmask, vlhs, vrhs), dpp::select, cmask, vlhs,
                    vrhs);
                test::operation_fixture<A>::test(test::bitcmp,
                    dpp::select(mmask, vlhs, dpp::zero), dpp::select, cmask,
                    vlhs, dpp::zero);
                test::operation_fixture<A>::test(test::bitcmp,
                    dpp::select(mmask, dpp::zero, vlhs), dpp::select, cmask,
                    dpp::zero, vlhs);
                test::operation_fixture<A>::test(test::bitcmp,
                    dpp::select(mmask, vlhs, dpp::all_bits), dpp::select, cmask,
                    vlhs, dpp::all_bits);
                test::operation_fixture<A>::test(test::bitcmp,
                    dpp::select(mmask, dpp::all_bits, vlhs), dpp::select, cmask,
                    dpp::all_bits, vlhs);
            });
        }
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr void mask_run(Rng& engine) {
        test::mask_generator<A, E> const mask_generator;
        for (auto const i : test_count<E>()) {
            auto const mlhs = mask_generator(engine);
            auto const mrhs = mask_generator(engine);
            auto const mmask = mask_generator(engine);
            constexpr bool negatable = requires {
                { !mmask } -> dpp::simd_mask;
            };

            auto mexpected =
                dpp::bwor(dpp::bwand(mlhs, mmask), dpp::bwandnot(mrhs, mmask));
            assert(dpp::all_of(
                dpp::cmpeq(mexpected, dpp::select(mmask, mlhs, mrhs))));
            if constexpr (negatable)
                assert(dpp::all_of(
                    dpp::cmpeq(mexpected, dpp::select(!mmask, mrhs, mlhs))));

            mexpected = dpp::bwand(mlhs, mmask);
            assert(dpp::all_of(
                dpp::cmpeq(mexpected, dpp::select(mmask, mlhs, dpp::zero))));
            if constexpr (negatable)
                assert(dpp::all_of(dpp::cmpeq(
                    mexpected, dpp::select(!mmask, dpp::zero, mlhs))));

            mexpected = dpp::bwandnot(mlhs, mmask);
            assert(dpp::all_of(
                dpp::cmpeq(mexpected, dpp::select(mmask, dpp::zero, mlhs))));
            if constexpr (negatable)
                assert(dpp::all_of(dpp::cmpeq(
                    mexpected, dpp::select(!mmask, mlhs, dpp::zero))));

            mexpected = dpp::bwornot(mlhs, mmask);
            assert(dpp::all_of(dpp::cmpeq(
                mexpected, dpp::select(mmask, mlhs, dpp::all_bits))));
            if constexpr (negatable)
                assert(dpp::all_of(dpp::cmpeq(
                    mexpected, dpp::select(!mmask, dpp::all_bits, mlhs))));

            mexpected = dpp::bwor(mlhs, mmask);
            assert(dpp::all_of(dpp::cmpeq(
                mexpected, dpp::select(mmask, dpp::all_bits, mlhs))));
            if constexpr (negatable)
                assert(dpp::all_of(dpp::cmpeq(
                    mexpected, dpp::select(!mmask, mlhs, dpp::all_bits))));
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const mlhs = mask_generator(engine);
            auto const mrhs = mask_generator(engine);
            run_const_mask_test<E>([&](auto cmask) {
                auto const mmask =
                    dpp::from_bitset<A, E>(dpp::to_bitset(cmask));
                test::operation_fixture<A>::test(dpp::select(mmask, mlhs, mrhs),
                    dpp::select, cmask, mlhs, mrhs);
                test::operation_fixture<A>::test(
                    dpp::select(mmask, mlhs, dpp::zero), dpp::select, cmask,
                    mlhs, dpp::zero);
                test::operation_fixture<A>::test(
                    dpp::select(mmask, dpp::zero, mlhs), dpp::select, cmask,
                    dpp::zero, mlhs);
                test::operation_fixture<A>::test(
                    dpp::select(mmask, mlhs, dpp::all_bits), dpp::select, cmask,
                    mlhs, dpp::all_bits);
                test::operation_fixture<A>::test(
                    dpp::select(mmask, dpp::all_bits, mlhs), dpp::select, cmask,
                    dpp::all_bits, mlhs);
            });
        }
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return selection::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        vector_run<E>(engine);
        mask_run<E>(engine);
        return true;
    }
};

} // namespace dpl::test
