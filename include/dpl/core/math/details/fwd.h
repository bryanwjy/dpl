// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
DPL_EXPORT namespace dx = __DPL datapar; // NOLINT
} // namespace datapar::fmath

namespace datapar::internal {
DPL_EXPORT namespace mx = __DPL datapar::fmath; // NOLINT
}

namespace datapar::rounding {
DPL_EXPORT namespace mx = __DPL datapar::fmath; // NOLINT
}

namespace datapar::fpfix {
DPL_EXPORT namespace mx = __DPL datapar::fmath; // NOLINT
}
DPL_DEFAULT_NAMESPACE_END
