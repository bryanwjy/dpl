// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_base_of.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
struct compliance_t {
    __DPL_HIDE_FROM_ABI constexpr ~compliance_t() = default;
};

template <typename T>
concept compliance_tag = is_base_of_v<fmath::compliance_t, T>;

namespace compliance {

struct full_t : private fmath::compliance_t {
    __DPL_HIDE_FROM_ABI explicit constexpr full_t() noexcept = default;
};

struct speed_t : public full_t {
    __DPL_HIDE_FROM_ABI explicit constexpr speed_t() noexcept = default;
};

struct unsafe_t : public speed_t {
    __DPL_HIDE_FROM_ABI explicit constexpr unsafe_t() noexcept = default;
};

inline constexpr unsafe_t unsafe{};
inline constexpr speed_t speed{};
inline constexpr full_t full{};
} // namespace compliance
} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END
