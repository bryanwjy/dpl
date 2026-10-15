// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.sad;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class sad {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;
    template <typename E>
    using vec_t = dpp::make_canonical_vector_t<E, A>;

    template <typename To, typename E>
    static constexpr To absdiff(E lhs, E rhs) noexcept {
        static_assert((dpl::is_signed_v<E> && dpl::is_signed_v<To>) ||
            (dpl::is_unsigned_v<E> && dpl::is_unsigned_v<To>));
        if constexpr (dpl::is_signed_v<E>) {
            auto const diff = static_cast<To>(lhs) - static_cast<To>(rhs);
            return diff < 0 ? -diff : diff;
        } else {
            return lhs < rhs ? rhs - lhs : lhs - rhs;
        }
    }

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
    static constexpr bool run_all(dpl::type_pack<Es...> from, Rng& engine) {
        // Test is only for integrals
        static_assert((... && integral<Es>));
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E>) {
                return [&]<size_t I>(this auto self, dpl::size_constant<I>) {
                    if constexpr (I <= dpl::type_bit_v<uint64>) {
                        auto result = false;
                        if constexpr (dpl::is_signed_v<E>) {
                            using To = dpl::signed_integral_type_t<I>;
                            result = sad::template run<E, To>(engine);
                        } else {
                            using To = dpl::unsigned_integral_type_t<I>;
                            result = sad::template run<E, To>(engine);
                        }
                        return result && self(dpl::size_constant<(I * 2)>{});
                    } else {
                        return true;
                    }
                }(dpl::size_constant<dpl::type_bit_v<E>>{});
            },
            from);
    }

    template <dpp::simd_element_for<A> E, dpp::simd_element_for<A> To,
        rng_like Rng>
    static constexpr bool run(Rng& engine) {
        auto const data_generator = []() {
            if constexpr (dpl::is_same_v<E, To>) {
                constexpr auto rangemin =
                    dpl::integral_traits<E>::min_value / 4;
                constexpr auto rangemax =
                    dpl::integral_traits<E>::max_value / 4;
                return test::array_generator<A, E>(rangemin, rangemax);
            } else {
                return test::array_generator<A, E>{};
            }
        }();
        test::array_generator<A, To> const addend_generator(
            test::half_range<To>);
        test::mask_generator<A, To> const mask_generator;
        auto const vtrue = dpp::broadcast<To, A>(true);
        auto const vfalse = dpp::broadcast<To, A>(false);

        for (auto const _ : test_count<E>()) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const addend = addend_generator(engine);

            auto const expected = [&]() {
                auto result = addend;
                if constexpr (sizeof(E) == sizeof(To)) {
                    for (auto i = 0zu; i < abi_traits<To>::size(); ++i) {
                        result[i] += absdiff<To>(lhs[i], rhs[i]);
                    }
                } else {
                    for (auto i = 0zu; i < abi_traits<To>::size(); ++i) {
                        auto const ratio =
                            abi_traits<E>::size() / abi_traits<To>::size();
                        for (auto j = 0zu; j < ratio; ++j) {
                            auto const idx = i * ratio + j;
                            result[i] += absdiff<To>(lhs[idx], rhs[idx]);
                        }
                    }
                }
                return result;
            }();

            operation_fixture<A>::test(expected, dpp::sad, addend, lhs, rhs);
            operation_fixture<A>::test_masked(
                dpp::sad, addend, vtrue, lhs, rhs);
            operation_fixture<A>::test_masked(
                dpp::sad, addend, vfalse, lhs, rhs);
            auto const mask = mask_generator(engine);
            operation_fixture<A>::test_masked(dpp::sad, addend, mask, lhs, rhs);
            operation_fixture<A>::test_masked(
                dpp::sad, dpp::zero, mask, addend, lhs, rhs);
        }

        if constexpr (dpp::fixed_width_abi<A>) {
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const addend = addend_generator(engine);
            run_const_mask_test<To>([&](auto cmask) {
                operation_fixture<A>::test_masked(
                    dpp::sad, addend, cmask, lhs, rhs);
                operation_fixture<A>::test_masked(
                    dpp::sad, dpp::zero, cmask, addend, lhs, rhs);
            });
        }

        return true;
    }
};
} // namespace dpl::test
