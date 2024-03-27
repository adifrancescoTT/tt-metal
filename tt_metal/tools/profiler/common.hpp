// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define HOST_SIDE_LOG "profile_log_host.csv"
#define DEVICE_SIDE_LOG "profile_log_device.csv"

namespace tt {

namespace tt_metal {

constexpr std::string_view PROFILER_RUNTIME_ROOT_DIR = "generated/profiler/";
constexpr std::string_view PROFILER_LOGS_DIR_NAME = ".logs/";

inline std::string PROFILER_ZONE_SRC_LOCATIONS_LOG = string(PROFILER_RUNTIME_ROOT_DIR) + string(PROFILER_LOGS_DIR_NAME) + "zone_src_locations.log";
}  // namespace tt_metal

}  // namespace tt
