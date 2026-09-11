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
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using mask_t = dpp::make_canonical_mask_t<E, A>;

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
        test::mask_generator<A, E> const mask_generator;
        auto const vtrue = dpp::broadcast<A, E>(true);
        auto const vfalse = dpp::broadcast<A, E>(false);
        auto const upper_count = abi_traits<E>::size() - abi_traits<E>::size();
        auto const upper = []() {
            if constexpr (sizeof(E) != 1 &&
                dpl::is_same_v<mask_t<E>, mask_t<int8>>) {
                return dpp::cmpge(
                    dpp::lane_index<int8, A>(), abi_traits<E>::size());
            } else {
                return nullptr;
            }
        }();

        for (auto const _ : test_count<E>()) {
            auto const data = data_generator(engine);
            auto const src = src_generator(engine);

            {
                auto const expected = [&]() {
                    auto result = src;
                    for (auto const i : linear_counter(result)) {
                        result[i] = dpl::countl_zero(dpl::to_unsigned(data[i]));
                    }
                    return result;
                }();

                test::operation_fixture<A>::test(
                    expected, dpp::countl_zero, data);
            }

            test::operation_fixture<A>::test_masked(
                dpp::countl_zero, src, vtrue, data);
            test::operation_fixture<A>::test_masked(
                dpp::countl_zero, src, vfalse, data);

            auto const mask = mask_generator(engine);
            test::operation_fixture<A>::test_masked(
                dpp::countl_zero, src, mask, data);
            test::operation_fixture<A>::test_masked(
                dpp::countl_zero, dpp::zero, mask, data);

            {
                auto ridx = src;
                for (auto const i : linear_counter(ridx)) {
                    auto const j = abi_traits<E>::size() - 1 - i;
                    ridx[j] = i;
                }

                dpp::store(
                    dpp::select(mask, dpp::load<A>(ridx.data()), dpp::all_bits),
                    ridx.data());

                auto const expected = [&]() -> count_t {
                    for (auto const i : linear_counter(ridx)) {
                        auto const j = abi_traits<E>::size() - 1 - i;
                        if (ridx[j] < abi_traits<E>::size()) {
                            return ridx[j];
                        }
                    }

                    return abi_traits<E>::size();
                }();

                if constexpr (sizeof(E) != 1 &&
                    dpl::is_same_v<mask_t<E>, mask_t<int8>>) {
                    assert(dpp::countl_zero(dpp::bwandnot(mask, upper)) ==
                        expected);
                } else {
                    assert(dpp::countl_zero(mask) == expected);
                }
            }
        }

        for (auto const _ : test_count<E>()) {
            auto const data = data_generator(engine);
            auto const src = src_generator(engine);
            {
                auto const expected = [&]() {
                    auto result = src;
                    for (auto const i : linear_counter(result)) {
                        result[i] = dpl::countl_one(dpl::to_unsigned(data[i]));
                    }
                    return result;
                }();

                test::operation_fixture<A>::test(
                    expected, dpp::countl_one, data);
            }
            test::operation_fixture<A>::test_masked(
                dpp::countl_one, src, vtrue, data);
            test::operation_fixture<A>::test_masked(
                dpp::countl_one, src, vfalse, data);

            auto const mask = mask_generator(engine);
            test::operation_fixture<A>::test_masked(
                dpp::countl_one, src, mask, data);
            test::operation_fixture<A>::test_masked(
                dpp::countl_one, dpp::zero, mask, data);

            {
                auto ridx = src;
                for (auto const i : linear_counter(ridx)) {
                    auto const j = abi_traits<E>::size() - 1 - i;
                    ridx[j] = i;
                }

                dpp::store(
                    dpp::select(mask, dpp::all_bits, dpp::load<A>(ridx.data())),
                    ridx.data());

                auto const expected = [&]() -> count_t {
                    for (auto const i : linear_counter(ridx)) {
                        auto const j = abi_traits<E>::size() - 1 - i;
                        if (ridx[j] < abi_traits<E>::size()) {
                            return ridx[j];
                        }
                    }

                    return abi_traits<E>::size();
                }();

                if constexpr (sizeof(E) != 1 &&
                    dpl::is_same_v<mask_t<E>, mask_t<int8>>) {
                    assert(dpp::countl_one(dpp::bwor(mask, upper)) == expected);
                } else {
                    assert(dpp::countl_one(mask) == expected);
                }
            }
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            run_const_mask_test<E>([&](auto cmask) {
                auto const data = data_generator(engine);
                auto const src = src_generator(engine);
                operation_fixture<A>::test_masked(
                    dpp::countl_zero, src, cmask, data);
                operation_fixture<A>::test_masked(
                    dpp::countl_zero, dpp::zero, cmask, data);

                operation_fixture<A>::test_masked(
                    dpp::countl_one, src, cmask, data);
                operation_fixture<A>::test_masked(
                    dpp::countl_one, dpp::zero, cmask, data);
            });
        }

        return true;
    }
};

} // namespace dpl::test
