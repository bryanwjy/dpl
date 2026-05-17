// Copyright 2025-2026 Bryan Wong

// GCC has a large default
// @dpl[clang].compile-flags: -fconstexpr-steps=2000000
// @dpl[msvc].compile-flags: /constexpr:steps2000000

#include <cassert>

import dpl.xmm;

namespace {
namespace xmm = dpl::datapar::xmm;
namespace dpp = dpl::datapar;
template <typename... Ts>
constexpr auto element_count = dpp::simd_abi_traits<Ts...>::size;
static_assert(dpp::basic_simd_type<
    dpl::datapar::basic_simd<signed char, dpl::datapar::xmm::abi_tag>>);

template <typename T>
consteval T repeat_as(unsigned char val) {
    T x = val;
    for (auto i = 1u; i < sizeof(T); ++i) {
        x = (x << dpl::char_bit_v) | val;
    }
    return x;
}
constexpr auto min(auto lhs, auto rhs) noexcept {
    return lhs < rhs ? lhs : rhs;
}

template <dpl::integral To, dpl::integral From>
constexpr bool test(From src) noexcept {
    constexpr auto keep =
        min(element_count<To, xmm::abi_tag>, element_count<From, xmm::abi_tag>);
    constexpr auto keep_mask = dpp::imm<(1 << keep) - 1>;

    return dpp::all_of(
        dpp::element_cast<To>(dpp::broadcast<From, xmm::abi_tag>(src)) ==
        dpp::bit_keep(
            keep_mask, dpp::broadcast<To, xmm::abi_tag>(static_cast<To>(src))));
}

template <dpl::integral From, dpl::integral To>
constexpr void test() noexcept {
    assert(test<To>(static_cast<From>(0)));
    assert(test<To>(static_cast<From>(1)));
    assert(test<To>(static_cast<From>(-1)));
    assert(test<To>(dpp::msb_v<From>));
    assert(test<To>(static_cast<From>(~dpp::msb_v<From>)));
    if constexpr (dpl::signed_integral<From>) {
        assert(test<To>(static_cast<From>(-1)));
        assert(test<To>(dpp::min_value_v<From>));
        assert(test<To>(dpp::max_value_v<From>));
        if constexpr (dpl::unsigned_integral<To>) {
            assert(test<To>(static_cast<From>(-2)));
        }
    } else {
        assert(test<To>(dpp::max_value_v<From>));
        if constexpr (dpl::signed_integral<To>) {
            assert(test<To>(static_cast<From>(dpp::max_value_v<To>) + 1));
            assert(test<To>(static_cast<From>(-1)));
        }
    }

    constexpr auto dwidth = sizeof(To) * dpl::char_bit_v;

    if constexpr (sizeof(From) == sizeof(To)) {
        assert(test<To>(repeat_as<From>(0xAA)));
        assert(test<To>(repeat_as<From>(0x55)));
    } else if constexpr (sizeof(From) < sizeof(To)) {
        if constexpr (dwidth < 64) {
            assert(test<To>(static_cast<From>((1ll << dwidth) - 1)));
            assert(test<To>(static_cast<From>(1ll << dwidth)));
            assert(test<To>(static_cast<From>((1ll << dwidth) + 1)));
            if constexpr (dpl::signed_integral<From>) {
                assert(test<To>(-static_cast<From>((1ll << dwidth) - 1)));
                assert(test<To>(-static_cast<From>(1ll << dwidth)));
                assert(test<To>(-static_cast<From>((1ll << dwidth) + 1)));
            }
        }

        if constexpr (dpl::signed_integral<To>) {
            if constexpr (dpp::min_value_v<From> < dpp::min_value_v<To>) {
                assert(test<To>(static_cast<From>(dpp::min_value_v<To>)));
                assert(test<To>(static_cast<From>(dpp::min_value_v<To>) - 1));
            }

            if constexpr (dpp::max_value_v<From> > dpp::max_value_v<To>) {
                assert(test<To>(static_cast<From>(dpp::max_value_v<To>)));
                assert(test<To>(static_cast<From>(dpp::max_value_v<To>) + 1));
            }
        }

        if constexpr (sizeof(From) > sizeof(To)) {
            constexpr auto offset =
                (sizeof(From) - sizeof(To)) * dpl::char_bit_v;
            assert(test<To>(static_cast<From>(1ll << (dwidth + offset))));
            if constexpr (dpl::signed_integral<From>) {
                assert(test<To>(-static_cast<From>(1ll << (dwidth + offset))));
            }
        }
    }
}

constexpr bool test() noexcept {
    test<dpl::int8, dpl::int8>();
    test<dpl::int8, dpl::uint8>();
    test<dpl::int8, dpl::int16>();
    test<dpl::int8, dpl::uint16>();
    test<dpl::int8, dpl::int32>();
    test<dpl::int8, dpl::uint32>();
    test<dpl::int8, dpl::int64>();
    test<dpl::int8, dpl::uint64>();

    test<dpl::uint8, dpl::int8>();
    test<dpl::uint8, dpl::uint8>();
    test<dpl::uint8, dpl::int16>();
    test<dpl::uint8, dpl::uint16>();
    test<dpl::uint8, dpl::int32>();
    test<dpl::uint8, dpl::uint32>();
    test<dpl::uint8, dpl::int64>();
    test<dpl::uint8, dpl::uint64>();

    test<dpl::int16, dpl::int8>();
    test<dpl::int16, dpl::uint8>();
    test<dpl::int16, dpl::int16>();
    test<dpl::int16, dpl::uint16>();
    test<dpl::int16, dpl::int32>();
    test<dpl::int16, dpl::uint32>();
    test<dpl::int16, dpl::int64>();
    test<dpl::int16, dpl::uint64>();

    test<dpl::uint16, dpl::int8>();
    test<dpl::uint16, dpl::uint8>();
    test<dpl::uint16, dpl::int16>();
    test<dpl::uint16, dpl::uint16>();
    test<dpl::uint16, dpl::int32>();
    test<dpl::uint16, dpl::uint32>();
    test<dpl::uint16, dpl::int64>();
    test<dpl::uint16, dpl::uint64>();

    test<dpl::int32, dpl::int8>();
    test<dpl::int32, dpl::uint8>();
    test<dpl::int32, dpl::int16>();
    test<dpl::int32, dpl::uint16>();
    test<dpl::int32, dpl::int32>();
    test<dpl::int32, dpl::uint32>();
    test<dpl::int32, dpl::int64>();
    test<dpl::int32, dpl::uint64>();

    test<dpl::uint32, dpl::int8>();
    test<dpl::uint32, dpl::uint8>();
    test<dpl::uint32, dpl::int16>();
    test<dpl::uint32, dpl::uint16>();
    test<dpl::uint32, dpl::int32>();
    test<dpl::uint32, dpl::uint32>();
    test<dpl::uint32, dpl::int64>();
    test<dpl::uint32, dpl::uint64>();

    test<dpl::int64, dpl::int8>();
    test<dpl::int64, dpl::uint8>();
    test<dpl::int64, dpl::int16>();
    test<dpl::int64, dpl::uint16>();
    test<dpl::int64, dpl::int32>();
    test<dpl::int64, dpl::uint32>();
    test<dpl::int64, dpl::int64>();
    test<dpl::int64, dpl::uint64>();

    test<dpl::uint64, dpl::int8>();
    test<dpl::uint64, dpl::uint8>();
    test<dpl::uint64, dpl::int16>();
    test<dpl::uint64, dpl::uint16>();
    test<dpl::uint64, dpl::int32>();
    test<dpl::uint64, dpl::uint32>();
    test<dpl::uint64, dpl::int64>();
    test<dpl::uint64, dpl::uint64>();
    return true;
}
} // namespace

int main() {
    static_assert(test());
    assert(test());
    return 0;
}
