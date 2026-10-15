// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
export module dpl.test.harness.operations.element_cast:int_int;
export import dpl.test.support;

import dpl;

export namespace dpl::test {

namespace dpp = dpl::datapar;

template <dpp::simd_abi A>
class int_int_cast {
private:
    using abi_t = A;
    template <typename E>
    using abi_traits = dpp::simd_abi_traits<abi_t, E>;

    template <typename T>
    static consteval T repeat_byte_as(unsigned char val) {
        dpl::bitset<dpl::char_bit_v> const byte(val);
        return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
            return dpl::bit_cast<T>(dpl::bitset(((void)Is, byte)...));
        }(dpl::make_index_sequence<sizeof(T)>{});
    }

    template <dpl::integral To, dpl::integral From>
    static constexpr bool test(From src) noexcept {
        constexpr auto keep =
            dpp::min(abi_traits<To>::size(), abi_traits<From>::size());

        auto const actual =
            dpp::element_cast<To>(dpp::broadcast<From, abi_t>(src));
        if constexpr (dpp::fixed_width_abi<abi_t>) {
            constexpr auto M = (1 << keep) - 1;
            auto const expected = dpp::selecti<M>(
                dpp::broadcast<To, abi_t>(static_cast<To>(src)), dpp::zero);
            return dpp::all_of(actual == expected);
        } else {
            auto const expected =
                dpp::select(dpp::lane_index<To, abi_t>() < keep,
                    dpp::broadcast<To, abi_t>(static_cast<To>(src)), dpp::zero);
            return dpp::all_of(actual == expected);
        }
    }

    template <dpl::integral From, dpl::integral To>
    static constexpr void test() noexcept {
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

        constexpr auto dwidth = dpl::integral_traits<To>::width;

        if constexpr (sizeof(From) == sizeof(To)) {
            assert(test<To>(repeat_byte_as<From>(0xAA)));
            assert(test<To>(repeat_byte_as<From>(0x55)));
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
                    assert(
                        test<To>(static_cast<From>(dpp::min_value_v<To>) - 1));
                }

                if constexpr (dpp::max_value_v<From> > dpp::max_value_v<To>) {
                    assert(test<To>(static_cast<From>(dpp::max_value_v<To>)));
                    assert(
                        test<To>(static_cast<From>(dpp::max_value_v<To>) + 1));
                }
            }

            if constexpr (sizeof(From) > sizeof(To)) {
                constexpr auto offset = dpl::integral_traits<From>::width -
                    dpl::integral_traits<To>::width;
                assert(test<To>(static_cast<From>(1ll << (dwidth + offset))));
                if constexpr (dpl::signed_integral<From>) {
                    assert(
                        test<To>(-static_cast<From>(1ll << (dwidth + offset))));
                }
            }
        }
    }

public:
    static constexpr bool run_all() noexcept {
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
};

} // namespace dpl::test
