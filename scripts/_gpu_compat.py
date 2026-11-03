"""Keep scheduler UUID visibility compatible with vLLM 0.11's NVML queries.

CUDA accepts UUIDs, but vLLM 0.11 casts visibility entries to int when querying
NVML. Resolve UUIDs for those queries only; never rewrite CUDA_VISIBLE_DEVICES.
Entry modules also run this on multiprocessing spawn before vLLM imports.
"""

import os


def enable_vllm_uuid_queries():
    if not any(value.startswith("GPU-") for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")):
        return
    from vllm.platforms.interface import Platform
    if getattr(Platform, "_opd_uuid_queries", False):
        return
    original = Platform.device_id_to_physical_device_id.__func__

    def physical_id(cls, device_id):
        visible = os.environ.get(cls.device_control_env_var, "")
        if visible:
            value = visible.split(",")[device_id].strip()
            if value.startswith("GPU-"):
                from vllm.third_party import pynvml
                pynvml.nvmlInit()
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(value)
                    return pynvml.nvmlDeviceGetIndex(handle)
                finally:
                    pynvml.nvmlShutdown()
        return original(cls, device_id)

    Platform.device_id_to_physical_device_id = classmethod(physical_id)
    Platform._opd_uuid_queries = True
