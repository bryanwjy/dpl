// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
export module dpl.test:fp_int;

import dpl;
import :support.comparison;

export namespace dpl::test {

namespace dpp = dpl::datapar;

template <dpp::simd_abi A>
class fp_int_cast {
private:
    using abi_t = A;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<abi_t, E>;

    template <typename T, dpl::size_t N>
    struct array {
        T data[N];

        friend constexpr auto begin(array const& src) noexcept {
            return src.data;
        }

        friend constexpr auto end(array const& src) noexcept {
            return src.data + N;
        }

        static constexpr auto size() noexcept { return N; }
    };

    template <typename... Args>
    array(Args...)
        -> array<dpl::decay_t<dpl::common_type_t<Args...>>, sizeof...(Args)>;

    template <typename T>
    static constexpr auto make_array(
        dpl::convertible_to<T> auto... args) noexcept {
        return array<T, sizeof...(args)>{static_cast<T>(args)...};
    }

    template <dpl::integral I, dpl::floating_point_like F>
    static constexpr void general_int_to_fp() noexcept {
        auto const inputs =
            [](auto... vals) {
                constexpr I zero = 0;
                constexpr auto max_val = dpl::integral_traits<I>::max_value;
                constexpr auto min_val = dpl::integral_traits<I>::min_value;
                if constexpr (dpl::unsigned_integral<I>) {
                    return array{
                        (vals > max_val ? zero : static_cast<I>(vals))...};
                } else {
                    return array{
                        (vals > max_val ? zero : static_cast<I>(vals))...,
                        (-vals < min_val ? zero : static_cast<I>(-vals))...};
                }
            }(0, 1, 2, 3, 7, 15, 255, 1023, 65535, 1048575, 1234567, 7654321,
                10000000);
        auto const make_expected = [](I val) {
            constexpr auto keep =
                dpp::min(abi_traits<F>::size(), abi_traits<I>::size());
            auto const result = dpp::broadcast<F, abi_t>(static_cast<F>(val));
            if constexpr (dpp::fixed_width_abi<abi_t>) {
                constexpr auto M = (1 << keep) - 1;
                return dpp::selecti<M>(result, dpp::zero);
            } else {
                return dpp::select(
                    dpp::lane_index<F, abi_t>() < keep, result, dpp::zero);
            }
        };

        for (auto const val : inputs) {
            auto const in = dpp::broadcast<I, abi_t>(val);
            auto const expected = make_expected(val);
            auto const actual = dpp::element_cast<F>(in);
            assert(dpp::all_of(test::bitcmp(actual, expected)));
        }
    }

    template <dpl::integral I, dpl::floating_point_like F>
    static constexpr void parallel_int_to_fp() noexcept {
        auto const inputs = [](auto... vals) {
            constexpr I zero = 0;
            if constexpr (dpl::unsigned_integral<I>) {
                return array{static_cast<I>(vals)...};
            } else {
                return array{static_cast<I>(vals)..., static_cast<I>(-vals)...};
            }
        }(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        auto const make_expected = [](I const* ptr) {
            return [&]<dpl::size_t... Is>(dpl::index_sequence<Is...>) {
                constexpr auto keep =
                    dpp::min(abi_traits<F>::size(), abi_traits<I>::size());
                constexpr I zero = 0;
                return dpp::initialize<F, abi_t>(
                    static_cast<F>(Is < keep ? ptr[Is] : zero)...);
            }(dpl::make_index_sequence<abi_t::size / sizeof(F)>{});
        };

        assert((inputs.size() % abi_traits<I>::size()) == 0);
        for (auto i = 0; i < inputs.size(); i += abi_traits<I>::size()) {
            auto const* ptr = inputs.data + i;
            auto const in = dpp::load<abi_t>(ptr);
            auto const expected = make_expected(ptr);
            auto const actual = dpp::element_cast<F>(in);
            assert(dpp::all_of(test::bitcmp(actual, expected)));
        }
    }

    static constexpr bool large_int_to_sp() noexcept {
        constexpr auto make_expected = []<typename I>(I val) {
            constexpr auto keep = sizeof(I) > sizeof(float)
                ? abi_traits<float>::size() -
                    sizeof(I) /
                        sizeof(float) // NOLINT(bugprone-sizeof-expression)
                : abi_traits<float>::size();

            auto const result =
                dpp::broadcast<float, abi_t>(static_cast<float>(val));
            if constexpr (dpp::fixed_width_abi<abi_t>) {
                constexpr auto M = (1 << keep) - 1;
                return dpp::selecti<M>(result, dpp::zero);
            } else {
                return dpp::select(
                    dpp::lane_index<float, abi_t>() < keep, result, dpp::zero);
            }
        };

        constexpr auto test1 = [=]<dpl::integral T>(dpl::type_identity<T>) {
            // Tests for rounding issues from sligtly below and above 0x1.p24f
            // (16777216.0f)
            auto const src = make_array<T>(16777214, 16777215, 16777216,
                16777217, 16777218, 16777219, 16777220, 33554431, 33554432,
                33554433, 1073741824);
            for (auto const val : src) {
                auto const input = dpp::broadcast<T, abi_t>(val);
                auto const expected = make_expected(val);
                auto const actual = dpp::element_cast<float>(input);
                assert(dpp::all_of(test::bitcmp(actual, expected)));
            }

            if constexpr (dpl::signed_integral<T>) {
                for (auto const pval : src) {
                    auto const val = -pval;
                    auto const input = dpp::broadcast<T, abi_t>(val);
                    auto const expected = make_expected(val);
                    auto const actual = dpp::element_cast<float>(input);
                    assert(dpp::all_of(test::bitcmp(actual, expected)));
                }
            }
        };

        test1(dpl::type_identity<dpl::int32>{});
        test1(dpl::type_identity<dpl::int64>{});
        test1(dpl::type_identity<dpl::uint32>{});
        test1(dpl::type_identity<dpl::uint64>{});

        constexpr auto extra_large = [=]<dpl::integral T>(
                                         dpl::type_identity<T>) {
            // Extreme rounding
            array const src{1099511627776, 1125899906842624,
                1152921504606846976, 1152921504606846977, 1234567890123,
                9223372036854775807};
            for (auto const val : src) {
                auto const input = dpp::broadcast<T, abi_t>(val);
                auto const expected = make_expected(val);
                auto const actual = dpp::element_cast<float>(input);
                assert(dpp::all_of(test::bitcmp(actual, expected)));
            }

            if constexpr (dpl::signed_integral<T>) {
                for (auto const pval : src) {
                    auto const val = -pval;
                    auto const input = dpp::broadcast<T, abi_t>(val);
                    auto const expected = make_expected(val);
                    auto const actual = dpp::element_cast<float>(input);
                    assert(dpp::all_of(test::bitcmp(actual, expected)));
                }
            }
        };

        extra_large(dpl::type_identity<dpl::int64>{});
        extra_large(dpl::type_identity<dpl::uint64>{});

        return true;
    }

    static constexpr void test_int_to_fp() noexcept {
        using bfloat16 = dpl::ext::bfloat16;
        using float16 = dpl::ext::float16;

        general_int_to_fp<dpl::int64, float>();
        general_int_to_fp<dpl::int32, float>();
        general_int_to_fp<dpl::int16, float>();
        general_int_to_fp<dpl::int8, float>();

        general_int_to_fp<dpl::uint64, float>();
        general_int_to_fp<dpl::uint32, float>();
        general_int_to_fp<dpl::uint16, float>();
        general_int_to_fp<dpl::uint8, float>();

        parallel_int_to_fp<dpl::int64, float>();
        parallel_int_to_fp<dpl::int32, float>();
        parallel_int_to_fp<dpl::int16, float>();
        parallel_int_to_fp<dpl::int8, float>();

        parallel_int_to_fp<dpl::uint64, float>();
        parallel_int_to_fp<dpl::uint32, float>();
        parallel_int_to_fp<dpl::uint16, float>();
        parallel_int_to_fp<dpl::uint8, float>();

        general_int_to_fp<dpl::int64, bfloat16>();
        general_int_to_fp<dpl::int32, bfloat16>();
        general_int_to_fp<dpl::int16, bfloat16>();
        general_int_to_fp<dpl::int8, bfloat16>();

        general_int_to_fp<dpl::uint64, bfloat16>();
        general_int_to_fp<dpl::uint32, bfloat16>();
        general_int_to_fp<dpl::uint16, bfloat16>();
        general_int_to_fp<dpl::uint8, bfloat16>();

        parallel_int_to_fp<dpl::int64, bfloat16>();
        parallel_int_to_fp<dpl::int32, bfloat16>();
        parallel_int_to_fp<dpl::int16, bfloat16>();
        parallel_int_to_fp<dpl::int8, bfloat16>();

        parallel_int_to_fp<dpl::uint64, bfloat16>();
        parallel_int_to_fp<dpl::uint32, bfloat16>();
        parallel_int_to_fp<dpl::uint16, bfloat16>();
        parallel_int_to_fp<dpl::uint8, bfloat16>();

        // What about fp16?

        general_int_to_fp<dpl::int64, double>();
        general_int_to_fp<dpl::int32, double>();
        general_int_to_fp<dpl::int16, double>();
        general_int_to_fp<dpl::int8, double>();

        general_int_to_fp<dpl::uint64, double>();
        general_int_to_fp<dpl::uint32, double>();
        general_int_to_fp<dpl::uint16, double>();
        general_int_to_fp<dpl::uint8, double>();

        parallel_int_to_fp<dpl::int64, double>();
        parallel_int_to_fp<dpl::int32, double>();
        parallel_int_to_fp<dpl::int16, double>();
        parallel_int_to_fp<dpl::int8, double>();

        parallel_int_to_fp<dpl::uint64, double>();
        parallel_int_to_fp<dpl::uint32, double>();
        parallel_int_to_fp<dpl::uint16, double>();
        parallel_int_to_fp<dpl::uint8, double>();

        large_int_to_sp();
    }

    template <dpl::floating_point_like F, dpl::integral I>
    static constexpr void general_fp_to_int() noexcept {
        constexpr auto nextbefore = [](auto val) noexcept {
            using type = decltype(val);
            using rep = dpp::signed_representation_t<type>;
            return dpl::bit_cast<type>(
                static_cast<rep>(dpl::bit_cast<rep>(val) - 1));
        };
        auto const inputs = [](auto... vals) {
            if constexpr (dpl::unsigned_integral<I>) {
                return array{vals...};
            } else {
                return array{vals..., -vals...};
            }
        }(F(0.0), F(1.9), nextbefore(F(1)), dpp::min_value_v<F>);
        auto const make_expected = [](F val) {
            constexpr auto keep =
                dpp::min(abi_traits<F>::size(), abi_traits<I>::size());
            auto const result = dpp::broadcast<I, abi_t>(static_cast<I>(val));
            if constexpr (dpp::fixed_width_abi<abi_t>) {
                constexpr auto M = (1 << keep) - 1;
                return dpp::selecti<M>(result, dpp::zero);
            } else {
                return dpp::select(
                    dpp::lane_index<I, abi_t>() < keep, result, dpp::zero);
            }
        };

        for (auto const val : inputs) {
            auto const in = dpp::broadcast<F, abi_t>(val);
            auto const expected = make_expected(val);
            auto const actual = dpp::element_cast<I>(in);
            assert(dpp::all_of(expected == actual));
        }
    }

    template <dpl::floating_point_like F, dpl::integral I>
    static constexpr void large_fp_to_int() noexcept {
        static_assert(sizeof(I) == sizeof(float));
        using rep_t = dpp::signed_representation_t<F>;
        constexpr auto max = []() {
            // Gets the maximal value below which all integers are representable
            constexpr auto shift = dpl::floating_point_traits<F>::width -
                dpl::floating_point_traits<F>::digits - 1;
            auto const rep =
                dpl::to_signed(dpl::to_unsigned(rep_t(-1)) << shift);
            constexpr auto max_val = dpl::integral_traits<rep_t>::max_value;
            auto const val = rep & max_val;
            if (val > dpl::integral_traits<I>::max_value) {
                return static_cast<F>(dpl::integral_traits<I>::max_value);
            } else {
                return static_cast<F>(val);
            }
        }();
        auto const inputs = [](auto... vals) {
            if constexpr (dpl::unsigned_integral<I>) {
                return array{vals...};
            } else {
                return array{vals..., -vals..., F(dpp::min_value_v<I>)};
            }
        }(F(16777216.0), F(16777217.0), max);
        auto const make_expected = [](F val) {
            constexpr auto keep =
                dpp::min(abi_traits<F>::size(), abi_traits<I>::size());
            auto const result = dpp::broadcast<I, abi_t>(static_cast<I>(val));
            if constexpr (dpp::fixed_width_abi<abi_t>) {
                constexpr auto M = (1 << keep) - 1;
                return dpp::selecti<M>(result, dpp::zero);
            } else {
                return dpp::select(
                    dpp::lane_index<I, abi_t>() < keep, result, dpp::zero);
            }
        };

        for (auto const val : inputs) {
            auto const in = dpp::broadcast<F, abi_t>(val);
            auto const expected = make_expected(val);
            auto const actual = dpp::element_cast<I>(in);
            assert(dpp::all_of(expected == actual));
        }
    }

    template <dpl::floating_point_like F, dpl::integral I>
    static constexpr void large_fp_to_long() noexcept {
        using dpl::ext_literals::operator""_bf16;
        static_assert(sizeof(I) == sizeof(double));
        auto const inputs = []() {
            if constexpr (dpl::same_as<F, float>) {
                return [](auto... vals) {
                    if constexpr (dpl::unsigned_integral<I>) {
                        return array{vals...};
                    } else {
                        return array{vals..., -vals...};
                    }
                }(0x1.p24f, 0x1.p24f + 1.0f, 0x1.p40f, 0x1.p40f + 1.0f,
                           0x1.p56f);
            } else if constexpr (dpl::same_as<F, dpl::ext::bfloat16>) {
                return [](auto... vals) {
                    if constexpr (dpl::unsigned_integral<I>) {
                        return array{vals...};
                    } else {
                        return array{vals..., -vals...};
                    }
                }(0x1.p24_bf16, 0x1.p24_bf16 + 1.0_bf16, 0x1.p40_bf16,
                           0x1.p40_bf16 + 1.0_bf16, 0x1.p56_bf16);
            } else {
                static_assert(dpl::same_as<F, double>);
                return [](auto... vals) {
                    if constexpr (dpl::unsigned_integral<I>) {
                        return array{vals...};
                    } else {
                        return array{vals..., -vals..., -9223372036854775808.0};
                    }
                }(0x1.p53, 0x1.p53 + 1.0, 0x1.p60, 0x1.p60 + 1.0,
                           0x1.p60 + 128.0, 0x1.p60 + 255.0,
                           9223372036854773760.0);
            }
        }();
        auto const make_expected = [](F val) {
            return dpp::broadcast<I, abi_t>(static_cast<I>(val));
        };

        for (auto const val : inputs) {
            auto const in = dpp::broadcast<F, abi_t>(val);
            auto const expected = make_expected(val);
            auto const actual = dpp::element_cast<I>(in);
            assert(dpp::all_of(expected == actual));
        }
    }

    template <dpl::floating_point_like F, dpl::integral I>
    static constexpr void parallel_fp_to_int() noexcept {
        auto const inputs =
            [](auto... vals) {
                constexpr I zero = 0;
                if constexpr (dpl::unsigned_integral<I>) {
                    return array{vals...};
                } else {
                    return array{vals..., -vals...};
                }
            }(F(0), F(1), F(2), F(3), F(4), F(5), F(6), F(7), F(8), F(9), F(10),
                F(11), F(12), F(13), F(14), F(15));
        auto const make_expected = [](F const* ptr) {
            return [&]<dpl::size_t... Is>(dpl::index_sequence<Is...>) {
                constexpr auto keep =
                    dpp::min(abi_traits<F>::size(), abi_traits<I>::size());

                constexpr F zero = 0;
                return dpp::initialize<I, abi_t>(
                    static_cast<I>(Is < keep ? ptr[Is] : zero)...);
            }(dpl::make_index_sequence<abi_t::size / sizeof(I)>{});
        };

        assert((inputs.size() % abi_traits<F>::size()) == 0);
        for (auto i = 0; i < inputs.size(); i += abi_traits<F>::size()) {
            auto const* ptr = inputs.data + i;
            auto const in = dpp::load<abi_t>(ptr);
            auto const expected = make_expected(ptr);
            auto const actual = dpp::element_cast<I>(in);
            assert(dpp::all_of(expected == actual));
        }
    }

    static constexpr void test_fp_to_int() noexcept {
        using bfloat16 = dpl::ext::bfloat16;
        using float16 = dpl::ext::float16;

        general_fp_to_int<float, dpl::int64>();
        general_fp_to_int<float, dpl::int32>();
        general_fp_to_int<float, dpl::int16>();
        general_fp_to_int<float, dpl::int8>();
        general_fp_to_int<float, dpl::uint64>();
        general_fp_to_int<float, dpl::uint32>();
        general_fp_to_int<float, dpl::uint16>();
        general_fp_to_int<float, dpl::uint8>();

        general_fp_to_int<bfloat16, dpl::int64>();
        general_fp_to_int<bfloat16, dpl::int32>();
        general_fp_to_int<bfloat16, dpl::int16>();
        general_fp_to_int<bfloat16, dpl::int8>();
        general_fp_to_int<bfloat16, dpl::uint64>();
        general_fp_to_int<bfloat16, dpl::uint32>();
        general_fp_to_int<bfloat16, dpl::uint16>();
        general_fp_to_int<bfloat16, dpl::uint8>();

        general_fp_to_int<float16, dpl::int64>();
        general_fp_to_int<float16, dpl::int32>();
        general_fp_to_int<float16, dpl::int16>();
        general_fp_to_int<float16, dpl::int8>();
        general_fp_to_int<float16, dpl::uint64>();
        general_fp_to_int<float16, dpl::uint32>();
        general_fp_to_int<float16, dpl::uint16>();
        general_fp_to_int<float16, dpl::uint8>();

        general_fp_to_int<double, dpl::int64>();
        general_fp_to_int<double, dpl::int32>();
        general_fp_to_int<double, dpl::int16>();
        general_fp_to_int<double, dpl::int8>();
        general_fp_to_int<double, dpl::uint64>();
        general_fp_to_int<double, dpl::uint32>();
        general_fp_to_int<double, dpl::uint16>();
        general_fp_to_int<double, dpl::uint8>();

        large_fp_to_int<float, dpl::int32>();
        large_fp_to_int<float, dpl::uint32>();
        large_fp_to_long<float, dpl::int64>();
        large_fp_to_long<float, dpl::uint64>();
        large_fp_to_int<double, dpl::int32>();
        large_fp_to_int<double, dpl::uint32>();
        large_fp_to_long<double, dpl::int64>();
        large_fp_to_long<double, dpl::uint64>();

        large_fp_to_int<bfloat16, dpl::int32>();
        large_fp_to_int<bfloat16, dpl::uint32>();
        large_fp_to_long<bfloat16, dpl::int64>();
        large_fp_to_long<bfloat16, dpl::uint64>();

        parallel_fp_to_int<float, dpl::int64>();
        parallel_fp_to_int<float, dpl::int32>();
        parallel_fp_to_int<float, dpl::int16>();
        parallel_fp_to_int<float, dpl::int8>();
        parallel_fp_to_int<float, dpl::uint64>();
        parallel_fp_to_int<float, dpl::uint32>();
        parallel_fp_to_int<float, dpl::uint16>();
        parallel_fp_to_int<float, dpl::uint8>();

        parallel_fp_to_int<bfloat16, dpl::int64>();
        parallel_fp_to_int<bfloat16, dpl::int32>();
        parallel_fp_to_int<bfloat16, dpl::int16>();
        parallel_fp_to_int<bfloat16, dpl::int8>();
        parallel_fp_to_int<bfloat16, dpl::uint64>();
        parallel_fp_to_int<bfloat16, dpl::uint32>();
        parallel_fp_to_int<bfloat16, dpl::uint16>();
        parallel_fp_to_int<bfloat16, dpl::uint8>();

        parallel_fp_to_int<float16, dpl::int64>();
        parallel_fp_to_int<float16, dpl::int32>();
        parallel_fp_to_int<float16, dpl::int16>();
        parallel_fp_to_int<float16, dpl::int8>();
        parallel_fp_to_int<float16, dpl::uint64>();
        parallel_fp_to_int<float16, dpl::uint32>();
        parallel_fp_to_int<float16, dpl::uint16>();
        parallel_fp_to_int<float16, dpl::uint8>();

        parallel_fp_to_int<double, dpl::int64>();
        parallel_fp_to_int<double, dpl::int32>();
        parallel_fp_to_int<double, dpl::int16>();
        parallel_fp_to_int<double, dpl::int8>();
        parallel_fp_to_int<double, dpl::uint64>();
        parallel_fp_to_int<double, dpl::uint32>();
        parallel_fp_to_int<double, dpl::uint16>();
        parallel_fp_to_int<double, dpl::uint8>();
    }

public:
    static constexpr bool run_all() noexcept {
        test_int_to_fp();
        test_fp_to_int();
        return true;
    }
};

} // namespace dpl::test
