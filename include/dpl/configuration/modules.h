/* Copyright 2023-2025 Bryan Wong */

#pragma once

#ifndef DPL_MODULES
#  define DPL_MODULES 0
#endif

#if DPL_MODULES
#  define DPL_EXPORT export
#  define DPL_EXPORT_BEGIN export {
#  define DPL_EXPORT_END }
#else
#  define DPL_EXPORT
#  define DPL_EXPORT_BEGIN
#  define DPL_EXPORT_END
#endif
