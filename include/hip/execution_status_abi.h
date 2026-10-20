/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_EXECUTION_STATUS_ABI_H
#define HIP_EXECUTION_STATUS_ABI_H

#define HIPDNN_EP_ERR_DEVICE_EXECUTION_FAILED (-105)

#ifdef __cplusplus
#include <cstdio>

// Use only for execution API results, never argument-validation statuses.
inline int hipdnn_device_execution_status(int status) {
  if (status == 0 || status == HIPDNN_EP_ERR_DEVICE_EXECUTION_FAILED)
    return status;
  std::fprintf(stderr, "HIP device execution API failed with status %d\n",
               status);
  return HIPDNN_EP_ERR_DEVICE_EXECUTION_FAILED;
}
#endif

#endif
