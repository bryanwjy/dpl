// Copyright 2025-2026 Bryan Wong
#include <cassert>

import dpl.xmm;

namespace xmm = dpl::datapar::xmm;
namespace dpp = dpl::datapar;

namespace {

template <typename T, dpl::size_t N>
struct array {
    T data[N];

    friend constexpr auto begin(array const& src) noexcept { return src.data; }

    friend constexpr auto end(array const& src) noexcept {
        return src.data + N;
    }

    static constexpr auto size() noexcept { return N; }
};

template <typename... Args>
array(Args...)
    -> array<dpl::decay_t<dpl::common_type_t<Args...>>, sizeof...(Args)>;

template <dpl::floating_point T>
constexpr bool bit_equality(xmm::simd<T> lhs, xmm::simd<T> rhs) noexcept {
    return dpp::all_of(
        dpp::reinterpret<dpl::int32>(lhs) == dpp::reinterpret<dpl::int32>(rhs));
}

template <dpl::integral I, dpl::floating_point F>
constexpr void general_int_to_fp() noexcept {
    auto const inputs = [](auto... vals) {
        constexpr I zero = 0;
        if constexpr (dpl::unsigned_integral<I>) {
            return array{
                (vals > dpp::max_value_v<I> ? zero : static_cast<I>(vals))...};
        } else {
            return array{
                (vals > dpp::max_value_v<I> ? zero : static_cast<I>(vals))...,
                (-vals < dpp::min_value_v<I> ? zero
                                             : static_cast<I>(-vals))...};
        }
    }(0, 1, 2, 3, 7, 15, 255, 1023, 65535, 1048575, 1234567, 7654321, 10000000);
    auto const make_expected = [](I val) {
        constexpr auto to_keep = sizeof(I) > sizeof(F)
            ? dpp::element_count<F, xmm::abi_tag> -
                sizeof(I) / sizeof(F) // NOLINT(bugprone-sizeof-expression)
            : dpp::element_count<F, xmm::abi_tag>;
        constexpr auto mask = dpp::imm<(1 << to_keep) - 1>;
        auto const result =
            dpp::broadcast<F, xmm::abi_tag>(static_cast<F>(val));
        return dpp::bit_keep(mask, result);
    };

    for (auto const val : inputs) {
        auto const in = dpp::broadcast<I, xmm::abi_tag>(val);
        auto const expected = make_expected(val);
        auto const actual = dpp::cast<F>(in);
        assert(bit_equality(actual, expected));
    }
}

template <dpl::integral I, dpl::floating_point F>
constexpr void parallel_int_to_fp() noexcept {
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
            constexpr auto to_keep = sizeof(I) > sizeof(F)
                ? dpp::element_count<F, xmm::abi_tag> -
                    sizeof(I) / sizeof(F) // NOLINT(bugprone-sizeof-expression)
                : dpp::element_count<F, xmm::abi_tag>;

            constexpr I zero = 0;
            return dpp::initialize<F, xmm::abi_tag>(
                static_cast<F>(Is < to_keep ? ptr[Is] : zero)...);
        }(dpp::iota_sequence<F, xmm::abi_tag>);
    };

    assert((inputs.size() % dpp::element_count<I, xmm::abi_tag>) == 0);
    for (auto i = 0; i < inputs.size();
        i += dpp::element_count<I, xmm::abi_tag>) {
        auto const* ptr = inputs.data + i;
        auto const in = dpp::load<xmm::abi_tag>(ptr);
        auto const expected = make_expected(ptr);
        auto const actual = dpp::cast<F>(in);
        assert(bit_equality(actual, expected));
    }
}

constexpr bool general_int_to_fp() noexcept {
    general_int_to_fp<dpl::int64, float>();
    general_int_to_fp<dpl::int32, float>();
    general_int_to_fp<dpl::int16, float>();
    general_int_to_fp<dpl::int8, float>();

    general_int_to_fp<dpl::int64, double>();
    general_int_to_fp<dpl::int32, double>();
    general_int_to_fp<dpl::int16, double>();
    general_int_to_fp<dpl::int8, double>();

    general_int_to_fp<dpl::uint64, float>();
    general_int_to_fp<dpl::uint32, float>();
    general_int_to_fp<dpl::uint16, float>();
    general_int_to_fp<dpl::uint8, float>();

    general_int_to_fp<dpl::uint64, double>();
    general_int_to_fp<dpl::uint32, double>();
    general_int_to_fp<dpl::uint16, double>();
    general_int_to_fp<dpl::uint8, double>();

    return true;
}

constexpr bool parallel_int_to_fp() noexcept {
    parallel_int_to_fp<dpl::int64, float>();
    parallel_int_to_fp<dpl::int32, float>();
    parallel_int_to_fp<dpl::int16, float>();
    parallel_int_to_fp<dpl::int8, float>();

    parallel_int_to_fp<dpl::int64, double>();
    parallel_int_to_fp<dpl::int32, double>();
    parallel_int_to_fp<dpl::int16, double>();
    parallel_int_to_fp<dpl::int8, double>();

    parallel_int_to_fp<dpl::uint64, float>();
    parallel_int_to_fp<dpl::uint32, float>();
    parallel_int_to_fp<dpl::uint16, float>();
    parallel_int_to_fp<dpl::uint8, float>();

    parallel_int_to_fp<dpl::uint64, double>();
    parallel_int_to_fp<dpl::uint32, double>();
    parallel_int_to_fp<dpl::uint16, double>();
    parallel_int_to_fp<dpl::uint8, double>();

    return true;
}

template <dpl::floating_point F, dpl::integral I>
constexpr void general_fp_to_int() noexcept {
    auto const inputs = [](auto... vals) {
        if constexpr (dpl::unsigned_integral<I>) {
            return array{vals...};
        } else {
            return array{vals..., -vals...};
        }
    }(F(0.0), F(1.9), F(0.99999994), dpp::min_value_v<F>);
    auto const make_expected = [](F val) {
        constexpr auto to_keep = sizeof(I) > sizeof(F)
            ? dpp::element_count<I, xmm::abi_tag>
            : dpp::element_count<F, xmm::abi_tag>;
        constexpr auto mask = dpp::imm<(1 << to_keep) - 1>;
        auto const result =
            dpp::broadcast<I, xmm::abi_tag>(static_cast<I>(val));
        return dpp::bit_keep(mask, result);
    };

    for (auto const val : inputs) {
        auto const in = dpp::broadcast<F, xmm::abi_tag>(val);
        auto const expected = make_expected(val);
        auto const actual = dpp::cast<I>(in);
        assert(dpp::all_of(expected == actual));
    }
}

template <dpl::floating_point F, dpl::integral I>
constexpr void large_sp_to_int() noexcept {
    static_assert(sizeof(I) == sizeof(float));
    auto const inputs = [](auto... vals) {
        if constexpr (dpl::unsigned_integral<I>) {
            return array{vals...};
        } else {
            return array{vals..., -vals..., F(-2147483648.0)};
        }
    }(F(16777216.0), F(16777217.0), F(2147483520.0));
    auto const make_expected = [](F val) {
        constexpr auto to_keep = sizeof(I) > sizeof(F)
            ? dpp::element_count<I, xmm::abi_tag>
            : dpp::element_count<F, xmm::abi_tag>;
        constexpr auto mask = dpp::imm<(1 << to_keep) - 1>;
        auto const result =
            dpp::broadcast<I, xmm::abi_tag>(static_cast<I>(val));
        return dpp::bit_keep(mask, result);
    };

    for (auto const val : inputs) {
        auto const in = dpp::broadcast<F, xmm::abi_tag>(val);
        auto const expected = make_expected(val);
        auto const actual = dpp::cast<I>(in);
        assert(dpp::all_of(expected == actual));
    }
}

template <dpl::floating_point F, dpl::integral I>
constexpr void large_sp_to_long() noexcept {
    static_assert(sizeof(I) == sizeof(double));
    auto const inputs = []() {
        if constexpr (dpp::common_float_with<F, float>) {
            return [](auto... vals) {
                if constexpr (dpl::unsigned_integral<I>) {
                    return array{vals...};
                } else {
                    return array{vals..., -vals...};
                }
            }(0x1.p24f, 0x1.p24f + 1.0f, 0x1.p40f, 0x1.p40f + 1.0f, 0x1.p56f);
        } else {
            static_assert(dpp::common_float_with<F, double>);
            return [](auto... vals) {
                if constexpr (dpl::unsigned_integral<I>) {
                    return array{vals...};
                } else {
                    return array{vals..., -vals..., -9223372036854775808.0};
                }
            }(0x1.p53, 0x1.p53 + 1.0, 0x1.p60, 0x1.p60 + 1.0, 0x1.p60 + 128.0,
                       0x1.p60 + 255.0, 9223372036854773760.0);
        }
    }();
    auto const make_expected = [](F val) {
        return dpp::broadcast<I, xmm::abi_tag>(static_cast<I>(val));
    };

    for (auto const val : inputs) {
        auto const in = dpp::broadcast<F, xmm::abi_tag>(val);
        auto const expected = make_expected(val);
        auto const actual = dpp::cast<I>(in);
        assert(dpp::all_of(expected == actual));
    }
}

template <dpl::floating_point F, dpl::integral I>
constexpr void parallel_fp_to_int() noexcept {
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
            constexpr auto to_keep = sizeof(F) < sizeof(I)
                ? dpp::element_count<I, xmm::abi_tag>
                : dpp::element_count<F, xmm::abi_tag>;

            constexpr F zero = 0;
            return dpp::initialize<I, xmm::abi_tag>(
                static_cast<I>(Is < to_keep ? ptr[Is] : zero)...);
        }(dpp::iota_sequence<I, xmm::abi_tag>);
    };

    assert((inputs.size() % dpp::element_count<F, xmm::abi_tag>) == 0);
    for (auto i = 0; i < inputs.size();
        i += dpp::element_count<F, xmm::abi_tag>) {
        auto const* ptr = inputs.data + i;
        auto const in = dpp::load<xmm::abi_tag>(ptr);
        auto const expected = make_expected(ptr);
        auto const actual = dpp::cast<I>(in);
        assert(dpp::all_of(expected == actual));
    }
}

constexpr bool general_fp_to_int() noexcept {
    general_fp_to_int<float, dpl::int64>();
    general_fp_to_int<float, dpl::int32>();
    general_fp_to_int<float, dpl::int16>();
    general_fp_to_int<float, dpl::int8>();
    general_fp_to_int<float, dpl::uint64>();
    general_fp_to_int<float, dpl::uint32>();
    general_fp_to_int<float, dpl::uint16>();
    general_fp_to_int<float, dpl::uint8>();

    general_fp_to_int<double, dpl::int64>();
    general_fp_to_int<double, dpl::int32>();
    general_fp_to_int<double, dpl::int16>();
    general_fp_to_int<double, dpl::int8>();
    general_fp_to_int<double, dpl::uint64>();
    general_fp_to_int<double, dpl::uint32>();
    general_fp_to_int<double, dpl::uint16>();
    general_fp_to_int<double, dpl::uint8>();

    large_sp_to_int<float, dpl::int32>();
    large_sp_to_int<float, dpl::uint32>();
    large_sp_to_int<double, dpl::int32>();
    large_sp_to_int<double, dpl::uint32>();
    large_sp_to_long<float, dpl::int64>();
    large_sp_to_long<float, dpl::uint64>();
    large_sp_to_long<double, dpl::int64>();
    large_sp_to_long<double, dpl::uint64>();

    parallel_fp_to_int<float, dpl::int64>();
    parallel_fp_to_int<float, dpl::int32>();
    parallel_fp_to_int<float, dpl::int16>();
    parallel_fp_to_int<float, dpl::int8>();
    parallel_fp_to_int<float, dpl::uint64>();
    parallel_fp_to_int<float, dpl::uint32>();
    parallel_fp_to_int<float, dpl::uint16>();
    parallel_fp_to_int<float, dpl::uint8>();

    parallel_fp_to_int<double, dpl::int64>();
    parallel_fp_to_int<double, dpl::int32>();
    parallel_fp_to_int<double, dpl::int16>();
    parallel_fp_to_int<double, dpl::int8>();
    parallel_fp_to_int<double, dpl::uint64>();
    parallel_fp_to_int<double, dpl::uint32>();
    parallel_fp_to_int<double, dpl::uint16>();
    parallel_fp_to_int<double, dpl::uint8>();

    return true;
}

template <typename T>
constexpr auto make_array(dpl::convertible_to<T> auto... args) noexcept {
    return array<T, sizeof...(args)>{static_cast<T>(args)...};
}

constexpr bool large_int_to_sp() noexcept {
    constexpr auto make_expected = []<typename I>(I val) {
        constexpr auto to_keep = sizeof(I) > sizeof(float)
            ? dpp::element_count<float, xmm::abi_tag> -
                sizeof(I) / sizeof(float) // NOLINT(bugprone-sizeof-expression)
            : dpp::element_count<float, xmm::abi_tag>;
        constexpr auto mask = dpp::imm<(1 << to_keep) - 1>;
        auto const result =
            dpp::broadcast<float, xmm::abi_tag>(static_cast<float>(val));
        return dpp::bit_keep(mask, result);
    };

    constexpr auto test1 = [=]<dpl::integral T>(dpl::type_identity<T>) {
        // Tests for rounding issues from sligtly below and above 0x1.p24f
        // (16777216.0f)
        auto const src =
            make_array<T>(16777214, 16777215, 16777216, 16777217, 16777218,
                16777219, 16777220, 33554431, 33554432, 33554433, 1073741824);
        for (auto const val : src) {
            auto const input = dpp::broadcast<T, xmm::abi_tag>(val);
            auto const expected = make_expected(val);
            assert(bit_equality(dpp::cast<float>(input), expected));
        }

        if constexpr (dpl::signed_integral<T>) {
            for (auto const pval : src) {
                auto const val = -pval;
                auto const input = dpp::broadcast<T, xmm::abi_tag>(val);
                auto const expected = make_expected(val);
                assert(bit_equality(dpp::cast<float>(input), expected));
            }
        }
    };

    test1(dpl::type_identity<dpl::int32>{});
    test1(dpl::type_identity<dpl::int64>{});
    test1(dpl::type_identity<dpl::uint32>{});
    test1(dpl::type_identity<dpl::uint64>{});

    constexpr auto extra_large = [=]<dpl::integral T>(dpl::type_identity<T>) {
        // Extreme rounding
        array const src{1099511627776, 1125899906842624, 1152921504606846976,
            1152921504606846977, 1234567890123, 9223372036854775807};
        for (auto const val : src) {
            auto const input = dpp::broadcast<T, xmm::abi_tag>(val);
            auto const expected = make_expected(val);
            assert(bit_equality(dpp::cast<float>(input), expected));
        }

        if constexpr (dpl::signed_integral<T>) {
            for (auto const pval : src) {
                auto const val = -pval;
                auto const input = dpp::broadcast<T, xmm::abi_tag>(val);
                auto const expected = make_expected(val);
                assert(bit_equality(dpp::cast<float>(input), expected));
            }
        }
    };

    extra_large(dpl::type_identity<dpl::int64>{});
    extra_large(dpl::type_identity<dpl::uint64>{});

    return true;
}
} // namespace

int main() {
    static_assert(general_int_to_fp());
    assert(general_int_to_fp());
    static_assert(parallel_int_to_fp());
    assert(parallel_int_to_fp());
    static_assert(large_int_to_sp());
    assert(large_int_to_sp());
    static_assert(general_fp_to_int());
    assert(general_fp_to_int());
    return 0;
}
