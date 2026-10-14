# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Read the leader's current joints (torque off, nothing moves) and print them as a reset pose.

Usage: scripts/read_pose.sh (hold the leader at the pose, then run). Settings come from local.env through the
environment: LEADER_PORT, LEADER_ID, URDF, EE_BOUNDS_MIN, EE_BOUNDS_MAX.
"""

import json
import os

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader

leader = SOLeader(SO101LeaderConfig(port=os.environ["LEADER_PORT"], id=os.environ["LEADER_ID"]))
leader.connect(calibrate=False)
try:
    action = leader.get_action()
finally:
    leader.disconnect()

names = list(leader.bus.motors)
joints = np.array([action[f"{n}.pos"] for n in names])
ee = RobotKinematics(os.environ["URDF"], "gripper_frame_link", names).forward_kinematics(joints)[:3, 3]

bounds = {"min": json.loads(os.environ["EE_BOUNDS_MIN"]), "max": json.loads(os.environ["EE_BOUNDS_MAX"])}
inside = all(lo <= v <= hi for v, lo, hi in zip(ee, bounds["min"], bounds["max"], strict=True))

print("joints:", {n: round(float(v), 2) for n, v in zip(names, joints, strict=True)})
print("RESET_POSE / HOME_POSE =", json.dumps([round(float(v), 2) for v in joints]))
print("gripper tip xyz [m] =", [round(float(v), 3) for v in ee])
print("bounds min", bounds["min"], "max", bounds["max"])
print("INSIDE bounds" if inside else "OUTSIDE bounds -- move it into the box")
