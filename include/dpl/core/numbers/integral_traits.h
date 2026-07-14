// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/is_signed.h"
#  include "dpl/std/utility/bitset.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename>
struct integral_traits {};
// Two's complement integers; Integers must be two's complement >= C++20
template <integral T>
struct integral_traits<T> {
    using type = T;

    static constexpr auto radix = 2zu;

    static constexpr auto width = sizeof(T) * __DPL char_bit_v;

    static constexpr auto digits =
        width - static_cast<size_t>(__DPL is_signed_v<T>);

    static constexpr auto signbit = __DPL is_signed_v<T>
        ? ~bitset<width>() << (width - 1)
        : bitset<width>();

    static constexpr auto min_value = __DPL is_signed_v<T>
        ? static_cast<T>(__DPL to_unsigned(static_cast<T>(1)) << digits)
        : static_cast<T>(0);

    static constexpr auto max_value =
        static_cast<T>(~__DPL to_unsigned(min_value));
};

__DPL_DEFAULT_NAMESPACE_END
