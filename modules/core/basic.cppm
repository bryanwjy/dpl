// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.basic;
import :core.fwd;
import :core.concepts;
import :core.type_traits;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;

// IWYU pragma: begin_exports
#include "dpl/core/basic/aligned.h"
#include "dpl/core/basic/basic_simd.h"
#include "dpl/core/basic/broadcast.h"
#include "dpl/core/basic/broadcastable_base.h"
#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/immediate.h"
#include "dpl/core/basic/initialize.h"
#include "dpl/core/basic/load.h"
#include "dpl/core/basic/reinterpret.h"
#include "dpl/core/basic/simd_mask.h"
#include "dpl/core/basic/store.h"
#include "dpl/core/basic/to_basic_type.h"
#include "dpl/core/basic/to_simd_mask.h"
// IWYU pragma: end_exports
