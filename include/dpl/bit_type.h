#pragma once

#include "dpl/config.h"

#include "dpl/fwd.h"

#include "dpl/blob.h"

#if !DPL_MODULES
#  include <climits>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t Width>
requires (Width <= CHAR_BIT)
struct __DPL_PUBLIC_TEMPLATE bit_type {
    static_assert(N && !(N & (N - 1)));
    using type = unsigned char;
};
template <size_t Width>
requires (Width == 2 * CHAR_BIT)
struct __DPL_PUBLIC_TEMPLATE bit_type {
    using type = unsigned short;
};
template <size_t Width>
requires (Width == 4 * CHAR_BIT)
struct __DPL_PUBLIC_TEMPLATE bit_type {
    using type = unsigned int;
};
template <size_t Width>
requires (Width == 8 * CHAR_BIT)
struct __DPL_PUBLIC_TEMPLATE bit_type {
    using type = unsigned long long;
};

#if DPL_SUPPORTS_INT128

template <size_t Width>
requires (Width == 16 * CHAR_BIT)
struct __DPL_PUBLIC_TEMPLATE bit_type {
    using type = __uint128_t;
};

#endif

template <size_t Width>
struct __DPL_PUBLIC_TEMPLATE bit_type {
    using type = blob<Width / CHAR_BIT>;
};

DPL_DEFAULT_NAMESPACE_END
