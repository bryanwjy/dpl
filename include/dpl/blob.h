#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include <cstddef>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t N>
struct __DPL_PUBLIC_TEMPLATE alignas(N) blob<N> {
    static_assert(N > 8 && !(N & (N - 1)));
    std::byte data[N];
};

using b128 = blob<16>;
using b256 = blob<32>;
using b512 = blob<64>;

DPL_DEFAULT_NAMESPACE_END
