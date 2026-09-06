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

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> from, Rng& engine) {
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
        test::array_generator<A, To> const addend_generator(test::half_range);
        for (auto i = 0zu; i < 5; ++i) {
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

            // TODO Fix assignment harness
            auto const vaddend = dpp::load<To, A>(addend.data());
            dpl::test::binary_transform<A>::template test<E>(
                lhs, rhs,
                [vaddend](auto vlhs, auto vrhs) {
                    return dpp::sad(vaddend, vlhs, vrhs);
                },
                expected);

            if constexpr (dpp::fixed_width_abi<A>) {
                constexpr auto lanes = abi_traits<To>::size();
                constexpr auto loop_count = []() {
                    auto max = lanes > dpl::type_bit_v<size_t>
                        ? 128
                        : static_cast<size_t>(
                              dpl::to_underlying(~dpl::bitset<lanes>()));
                    return max < 128 ? max : 128;
                }();

                test::bit_generator<lanes> mask_generator;
                for (auto i = 0zu; i < loop_count; ++i) {
                    auto mexpected = expected;
                    auto const mask = mask_generator(engine);
                    for (auto j = 0zu; j < mexpected.size(); ++j) {
                        if (!mask[j]) {
                            mexpected[j] = addend[j];
                        }
                    }

                    auto const vmask = dpp::from_bitset<A, To>(mask);
                    test::binary_transform<A>::template test<E>(
                        lhs, rhs,
                        [vaddend, vmask](auto vlhs, auto vrhs) {
                            return dpp::sad(vaddend, vmask, vlhs, vrhs);
                        },
                        mexpected, test::bitcmp);

                    for (auto j = 0zu; j < mexpected.size(); ++j) {
                        if (!mask[j]) {
                            mexpected[j] = 0;
                        }
                    }

                    test::binary_transform<A>::template test<E>(
                        lhs, rhs,
                        [vaddend, vmask](auto vlhs, auto vrhs) {
                            return dpp::sad(
                                dpp::zero, vmask, vaddend, vlhs, vrhs);
                        },
                        mexpected, test::bitcmp);

                    test::binary_transform<A>::template test<E>(
                        lhs, rhs,
                        [vaddend, vmask](auto vlhs, auto vrhs) {
                            return dpp::sad(vmask, vaddend, vlhs, vrhs);
                        },
                        mexpected, test::bitcmp);
                }

                using bitset_t = dpl::bitset<lanes>;
                constexpr auto const_count = 4zu;
                constexpr auto masks = [lanes]() {
                    return dpl::apply(
                        [lanes](auto... idx) {
                            test::mt19937 rng{};
                            dpl::test::bit_generator<lanes> bitgen;
                            return array<bitset_t, const_count>{
                                (dpl::ignore = idx, bitgen(rng))...};
                        },
                        dpl::make_index_sequence<const_count>{});
                }();
                dpl::pack::for_each(
                    [&]<size_t I>(dpl::size_constant<I>) {
                        auto mexpected = expected;
                        constexpr auto mask = masks[I];
                        using cmask_t = dpp::const_mask<mask.size(),
                            dpl::to_underlying(mask)>;

                        for (auto j = 0zu; j < mexpected.size(); ++j) {
                            if (!mask[j]) {
                                mexpected[j] = addend[j];
                            }
                        }

                        test::binary_transform<A>::template test<E>(
                            lhs, rhs,
                            [vaddend](auto vlhs, auto vrhs) {
                                return dpp::sad(vaddend, cmask_t{}, vlhs, vrhs);
                            },
                            mexpected, test::bitcmp);

                        for (auto j = 0zu; j < mexpected.size(); ++j) {
                            if (!mask[j]) {
                                mexpected[j] = 0;
                            }
                        }

                        test::binary_transform<A>::template test<E>(
                            lhs, rhs,
                            [vaddend](auto vlhs, auto vrhs) {
                                return dpp::sad(
                                    dpp::zero, cmask_t{}, vaddend, vlhs, vrhs);
                            },
                            mexpected, test::bitcmp);

                        test::binary_transform<A>::template test<E>(
                            lhs, rhs,
                            [vaddend](auto vlhs, auto vrhs) {
                                return dpp::sad(cmask_t{}, vaddend, vlhs, vrhs);
                            },
                            mexpected, test::bitcmp);
                    },
                    dpl::make_index_sequence<const_count>{});
            }
        }
        return true;
    }
};
} // namespace dpl::test
