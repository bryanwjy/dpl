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
    using abi_t = A;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<abi_t, E>;
    template <typename E>
    using vec_t = dpp::basic_vector<E, abi_t>;

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr void vector_run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        test::array_generator<abi_t, E> const data_generator;
        test::bit_generator<abi_traits<E>::size> const bit_generator;
        for (auto i = 0zu; i < 4; ++i) {

            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto const mask = bit_generator(engine);
            auto const vmask = dpp::from_bitset<abi_t>(mask);
            auto expected = lhs;
            for (auto i = 0zu; i < mask.size(); ++i) {
                if (!mask[i]) {
                    expected[i] = rhs[i];
                }
            }

            test::binary_transform<abi_t>::template test<E>(
                lhs, rhs,
                [vmask](vec_t<E> lhs, vec_t<E> rhs) noexcept {
                    return dpp::select(vmask, lhs, rhs);
                },
                expected, test::bitcmp);

            // Test negated mask
            test::binary_transform<abi_t>::template test<E>(
                lhs, rhs,
                [vmask](vec_t<E> lhs, vec_t<E> rhs) noexcept {
                    return dpp::select(!vmask, rhs, lhs);
                },
                expected, test::bitcmp);

            for (auto i = 0zu; i < mask.size(); ++i) {
                if (!mask[i]) {
                    expected[i] = 0;
                }
            }

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [vmask](vec_t<E> lhs) noexcept {
                    return dpp::select(vmask, lhs, dpp::zero);
                },
                expected, test::bitcmp);

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [vmask](vec_t<E> lhs) noexcept {
                    return dpp::select(!vmask, dpp::zero, lhs);
                },
                expected, test::bitcmp);

            for (auto i = 0zu; i < mask.size(); ++i) {
                if (mask[i]) {
                    expected[i] = 0;
                } else {
                    expected[i] = lhs[i];
                }
            }

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [vmask](vec_t<E> arg) noexcept {
                    return dpp::select(vmask, dpp::zero, arg);
                },
                expected, test::bitcmp);

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [vmask](vec_t<E> arg) noexcept {
                    return dpp::select(!vmask, arg, dpp::zero);
                },
                expected, test::bitcmp);

            for (auto i = 0zu; i < mask.size(); ++i) {
                if (!mask[i]) {
                    expected[i] = dpl::bit_cast<E>(~test::make_bitset_t<E>());
                } else {
                    expected[i] = lhs[i];
                }
            }

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [vmask](vec_t<E> lhs) noexcept {
                    return dpp::select(vmask, lhs, dpp::all_bits);
                },
                expected, test::bitcmp);

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [vmask](vec_t<E> lhs) noexcept {
                    return dpp::select(!vmask, dpp::all_bits, lhs);
                },
                expected, test::bitcmp);

            for (auto i = 0zu; i < mask.size(); ++i) {
                if (mask[i]) {
                    expected[i] = dpl::bit_cast<E>(~test::make_bitset_t<E>());
                } else {
                    expected[i] = lhs[i];
                }
            }

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [vmask](vec_t<E> arg) noexcept {
                    return dpp::select(vmask, dpp::all_bits, arg);
                },
                expected, test::bitcmp);

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [vmask](vec_t<E> arg) noexcept {
                    return dpp::select(!vmask, arg, dpp::all_bits);
                },
                expected, test::bitcmp);
        }

        {
            constexpr auto lanes = abi_traits<E>::size();
            using alt_cmask_t =
                dpp::const_mask<lanes, test::repeat_byte<lanes>(0x55)>;
            constexpr auto mask = dpp::to_bitset(alt_cmask_t());
            auto const lhs = data_generator(engine);
            auto const rhs = data_generator(engine);
            auto expected = lhs;
            for (auto i = 0zu; i < mask.size(); ++i) {
                if (!mask[i]) {
                    expected[i] = rhs[i];
                }
            }

            test::binary_transform<abi_t>::template test<E>(
                lhs, rhs,
                [](vec_t<E> lhs, vec_t<E> rhs) noexcept {
                    return dpp::select(alt_cmask_t(), lhs, rhs);
                },
                expected, test::bitcmp);

            for (auto i = 0zu; i < mask.size(); ++i) {
                if (!mask[i]) {
                    expected[i] = 0;
                }
            }

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [](vec_t<E> lhs) noexcept {
                    return dpp::select(alt_cmask_t(), lhs, dpp::zero);
                },
                expected, test::bitcmp);

            for (auto i = 0zu; i < mask.size(); ++i) {
                if (mask[i]) {
                    expected[i] = 0;
                } else {
                    expected[i] = lhs[i];
                }
            }

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [](vec_t<E> lhs) noexcept {
                    return dpp::select(alt_cmask_t(), dpp::zero, lhs);
                },
                expected, test::bitcmp);

            for (auto i = 0zu; i < mask.size(); ++i) {
                if (!mask[i]) {
                    expected[i] = dpl::bit_cast<E>(~test::make_bitset_t<E>());
                } else {
                    expected[i] = lhs[i];
                }
            }

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [](vec_t<E> lhs) noexcept {
                    return dpp::select(alt_cmask_t(), lhs, dpp::all_bits);
                },
                expected, test::bitcmp);

            for (auto i = 0zu; i < mask.size(); ++i) {
                if (mask[i]) {
                    expected[i] = dpl::bit_cast<E>(~test::make_bitset_t<E>());
                } else {
                    expected[i] = lhs[i];
                }
            }

            test::unary_transform<abi_t>::template test<E>(
                lhs,
                [](vec_t<E> lhs) noexcept {
                    return dpp::select(alt_cmask_t(), dpp::all_bits, lhs);
                },
                expected, test::bitcmp);
        }
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr void mask_run(Rng& engine)
    requires dpp::fixed_width_abi<A>
    {
        test::bit_generator<abi_traits<E>::size> const bit_generator;
        for (auto i = 0zu; i < 4; ++i) {
            auto const lhs = bit_generator(engine);
            auto const rhs = bit_generator(engine);
            auto const mask = bit_generator(engine);
            auto const vlhs = dpp::from_bitset<abi_t>(lhs);
            auto const vrhs = dpp::from_bitset<abi_t>(rhs);
            auto const vmask = dpp::from_bitset<abi_t>(mask);

            auto expected = (lhs & mask) | (rhs & ~mask);
            auto const vexpected = [&]() {
                return dpp::from_bitset<abi_t>(expected);
            };
            assert(dpp::all_of(dpp::select(vmask, vlhs, vrhs) == vexpected()));
            assert(dpp::all_of(dpp::select(!vmask, vrhs, vlhs) == vexpected()));

            expected = lhs & mask;

            assert(dpp::all_of(
                dpp::select(vmask, vlhs, dpp::zero) == vexpected()));
            assert(dpp::all_of(
                dpp::select(!vmask, dpp::zero, vlhs) == vexpected()));

            expected = lhs & ~mask;

            assert(dpp::all_of(
                dpp::select(vmask, dpp::zero, vlhs) == vexpected()));
            assert(dpp::all_of(
                dpp::select(!vmask, vlhs, dpp::zero) == vexpected()));

            expected = lhs | ~mask;

            assert(dpp::all_of(
                dpp::select(vmask, vlhs, dpp::all_bits) == vexpected()));
            assert(dpp::all_of(
                dpp::select(!vmask, dpp::all_bits, vlhs) == vexpected()));

            expected = lhs | mask;

            assert(dpp::all_of(
                dpp::select(vmask, dpp::all_bits, vlhs) == vexpected()));
            assert(dpp::all_of(
                dpp::select(!vmask, vlhs, dpp::all_bits) == vexpected()));
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
