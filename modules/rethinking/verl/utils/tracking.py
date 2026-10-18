# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Console metrics for the single main log."""

class Tracking:
    def __init__(self, project_name, experiment_name, default_backend, config=None):
        if list(default_backend) != ["console"]:
            raise ValueError("Only console tracking is included in OPD_lite")
        print(f"Training {project_name}/{experiment_name}", flush=True)

    def log(self, data, step, backend=None):
        values = " - ".join(f"{key}: {value}" for key, value in sorted(data.items()))
        print(f"step: {step} - {values}", flush=True)
