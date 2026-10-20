/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_EXECUTION_STATUS_H
#define HIPDNN_EP_EXECUTION_STATUS_H

#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdlib>

// Generated graph calls discard individual runtime return values. Preserve a
// failure in the session flag checked before inference returns to the caller.
class ExecutionStatusGuard {
public:
  explicit ExecutionStatusGuard(RuntimeState *state) : state_(state) {}

  ExecutionStatusGuard(const ExecutionStatusGuard &) = delete;
  ExecutionStatusGuard &operator=(const ExecutionStatusGuard &) = delete;

  ~ExecutionStatusGuard() {
    if (!state_ || status_ == 0)
      return;
    hipdnn_ep_state_record_execution_status(state_, status_);
    auto stream = static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state_));
    auto error = hipdnn_ep_state_get_error_flag_device_ptr(state_);
    if (!stream || !error ||
        hipMemsetAsync(error, 0xff, sizeof(int), stream) != hipSuccess) {
      std::fprintf(stderr, "Cannot record HIP runtime execution failure\n");
      std::abort();
    }
  }

  int Finish(int status) {
    status_ = status;
    return status;
  }

private:
  RuntimeState *state_;
  int status_ = -1;
};

#endif // HIPDNN_EP_EXECUTION_STATUS_H
