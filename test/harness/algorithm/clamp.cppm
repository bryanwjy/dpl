// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test.harness.algorithm.clamp;
export import dpl.test.support;

import dpl;

namespace dpl::test {
namespace dpp = dpl::datapar;

export template <dpp::simd_abi A>
class clamp {
private:
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <typename E>
    static constexpr E expected_op(E val, E min, E max) noexcept {
        val = val < min ? min : val;
        val = val > max ? max : val;
        return val;
    }

public:
    template <rng_like Rng, dpp::simd_element_for<A>... Es>
    static constexpr bool run_all(dpl::type_pack<Es...> pack, Rng& engine) {
        return dpl::pack::all_of(
            [&]<typename E>(dpl::type_identity<E> tp) {
                return clamp::template run<E>(engine);
            },
            pack);
    }

    template <dpp::simd_element_for<A> E, rng_like Rng>
    static constexpr bool run(Rng& engine) {
        test::array_generator<A, E> const data_generator;
        for (auto i = 0zu; i < 5; ++i) {
            auto minval = data_generator(engine);
            auto maxval = data_generator(engine);
            for (auto i = 0zu; i < minval.size(); ++i) {
                if (minval[i] > maxval[i]) {
                    dpl::ranges::swap(minval[i], maxval[i]);
                }
            }

            auto const val = data_generator(engine);
            auto const src_generator = data_generator.scalar(engine) < 0
                ? test::scalar_generator<E>(
                      dpp::min_value_v<E>,
                      [&minval]() {
                          auto val = minval[0];
                          for (auto i = 1; i < minval.size(); ++i) {
                              if (val < minval[i]) {
                                  val = minval[i];
                              }
                          }
                          return val;
                      }())
                : test::scalar_generator<E>(
                      [&maxval]() {
                          auto val = maxval[0];
                          for (auto i = 1; i < maxval.size(); ++i) {
                              if (val > maxval[i]) {
                                  val = maxval[i];
                              }
                          }
                          return val;
                      }(),
                      dpp::max_value_v<E>);

            auto const ssrc = src_generator(engine);

            auto expected = val;
            for (auto i = 0zu; i < expected.size(); ++i) {
                expected[i] = expected_op(val[i], minval[i], maxval[i]);
            }

            test::ternary_transform<A>::template test<E>(
                val, minval, maxval, dpp::clamp, expected, test::bitcmp);
            test::ternary_transform<A>::template test_masked<E>(
                val, minval, maxval, dpp::clamp, ssrc);

            if not consteval {
                constexpr auto lanes = abi_traits<E>::size();
                test::bit_generator<lanes> mask_generator;
                constexpr auto count = lanes < 128 ? lanes : 128;
                auto const src = data_generator(engine);
                for (auto i = 0; i < 128; ++i) {
                    auto const mask = mask_generator(engine);
                    for (auto i = 0zu; i < expected.size(); ++i) {
                        if (mask[i]) {
                            expected[i] =
                                expected_op(val[i], minval[i], maxval[i]);
                        } else {
                            expected[i] = src[i];
                        }
                    }

                    auto const vsrc = dpp::load<A, E>(src.data());
                    auto const vmask = dpp::from_bitset<A, E>(mask);
                    test::ternary_transform<A>::template test<E>(
                        val, minval, maxval,
                        [&](auto vval, auto vmin, auto vmax) {
                            return dpp::clamp(vsrc, vmask, vval, vmin, vmax);
                        },
                        expected, test::bitcmp);

                    for (auto i = 0zu; i < expected.size(); ++i) {
                        if (!mask[i]) {
                            expected[i] = 0;
                        }
                    }

                    test::ternary_transform<A>::template test<E>(
                        val, minval, maxval,
                        [&](auto vval, auto vmin, auto vmax) {
                            return dpp::clamp(
                                dpp::zero, vmask, vval, vmin, vmax);
                        },
                        expected, test::bitcmp);
                }
            }
        }
        return true;
    }
};
} // namespace dpl::test
