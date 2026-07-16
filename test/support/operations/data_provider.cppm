// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>
#include <cstdlib>

export module dpl.test:data_provider;
import dpl;
import :utils.array;
import :utils.unique_array;

namespace dpl::test {

namespace dpp = dpl::datapar;
export template <dpp::simd_abi A>
class data_provider {
    template <dpp::simd_element_for<A> E>
    using abi_traits = dpp::simd_abi_traits<A, E>;

    template <dpp::simd_element_for<A> E>
    requires dpp::fixed_width_abi<A>
    static constexpr auto iota =
        dpl::make_index_sequence<abi_traits<E>::size>{};

    template <dpp::simd_element_for<A> E>
    using array_t = conditional_t<dpp::fixed_width_abi<A>,
        array<E, abi_traits<E>::size>, unique_array<E>>;

public:
    static inline dpl::uint64 rand_bits(size_t bits) {
        dpl::uint64 r = 0;
        auto const mask =
            bits == 64 ? ~r : ((static_cast<dpl::uint64>(1) << bits) - 1);
        while (bits > 0) {
            r = (r << 15) | (::rand() & 0x7fff); // portable: RAND_MAX >= 32767
            bits = bits > 15 ? bits - 15 : 0;
        }
        return r & mask;
    }

    template <dpp::simd_element_for<A> E>
    static inline E rand() noexcept {
        if constexpr (dpl::floating_point_like<E>) {
            static_assert(dpl::floating_point_traits<E>::has_hidden_bit);
            constexpr auto mantissa_count =
                dpl::floating_point_traits<E>::digits - 1;
            constexpr auto exp_count =
                dpl::floating_point_traits<E>::width - mantissa_count - 1;

            auto const sign = static_cast<uint64>(rand_bits(1zu));

            auto const exp = [=]() {
                auto const result = rand_bits(exp_count);
                constexpr auto width = dpl::floating_point_traits<E>::width;
                return result -
                    (dpl::bitset<width>(result) ==
                        dpl::floating_point_traits<E>::exponent_mask);
            }();

            auto const mantissa = rand_bits(mantissa_count);

            return dpl::bit_cast<E>(
                static_cast<dpp::unsigned_representation_t<E>>(
                    (sign << (exp_count + mantissa_count)) |
                    (exp << mantissa_count) | mantissa));
        } else {
            return static_cast<E>(rand_bits(sizeof(E) * dpl::char_bit_v));
        }
    }

    template <dpp::simd_element_for<A> E>
    static constexpr array_t<E> array() noexcept {
        if constexpr (dpp::fixed_width_abi<A>) {
            if consteval {
                return []<size_t... Is>(dpl::index_sequence<Is...>) {
                    return array_t<E>{static_cast<E>((Is % 2zu == 0)
                            ? static_cast<E>(Is + 1)
                            : -static_cast<E>(Is + 1))...};
                }(iota<E>);
            } else {
                return []<size_t... Is>(dpl::index_sequence<Is...>) {
                    return array_t<E>{(dpl::ignore = Is, rand<E>())...};
                }(iota<E>);
            }
        } else {
            array_t<E> data(abi_traits<E>::size());
            for (auto& val : data) {
                val = rand<E>();
            }
            return data;
        }
    }

    template <dpp::simd_element_for<A> E>
    static constexpr E src() noexcept {
        if consteval {
            return static_cast<E>(37);
        } else {
            return rand<E>();
        }
    }
};

} // namespace dpl::test
