#!/usr/bin/env python3
import copy
import os
import time
import torch
from std_msgs.msg import String, Bool
import csv
from scipy.interpolate import interp1d
import yaml


class Trajectory:
    """Interpolates a CSV trajectory for multiple joints."""

    def __init__(self, trajectory_path, env, env_ids, mirror=False):
        """Initialize a Trajectory from a CSV file at trajectory_path.
        if it's getup trajectory append the desired final pose so the robot is ready for next action
        expects rectangular shape for csv table"""
        self.mirror = mirror
        self.splines = {}
        self.step_map = {}
        self.time_to_last_pose = 1  # seconds
        self.env = env
        self.env_ids = env_ids
        with open(os.path.join(os.getcwd(), 'cfg/task/bez_kick_test.yaml'), 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.SafeLoader)

        with open(trajectory_path) as f:
            csv_traj = csv.reader(f)
            for row in csv_traj:
                joint_name = row[0]
                if joint_name == 'comment':
                    continue
                if joint_name == 'time':
                    self.times = list(map(float, row[1:]))
                    self.times = [0] + self.times + [self.times[-1] + self.time_to_last_pose]
                    self.max_time = self.times[-1]
                else:
                    joint_values = list(map(float, row[1:]))

                    last_pose_value = float(self.cfg["env"]["readyJointAngles"][joint_name])
                    # last_pose_value = 0.0
                    joint_values = [last_pose_value] + joint_values + [last_pose_value]
                    self.splines[joint_name] = interp1d(self.times, joint_values)

    def get_setpoint(self, timestamp):
        """Get the position of each joint at timestamp.
        If timestamp < 0 or timestamp > self.total_time this will throw a ValueError.
        """
        return {joint: spline(timestamp) for joint, spline in self.splines.items()}

    def joints(self):
        """Returns a list of joints in this trajectory."""
        return self.splines.keys()

    def publish(self):

        t = 0
        print(self.max_time)
        while t < self.max_time:

            name = ["head_motor_0", "head_motor_1",
                    "left_arm_motor_0", "left_arm_motor_1",
                    "left_leg_motor_0", "left_leg_motor_1","left_leg_motor_2", "left_leg_motor_3", "left_leg_motor_4", "left_leg_motor_5",
                    "right_arm_motor_0", "right_arm_motor_1",
                    "right_leg_motor_0", "right_leg_motor_1", "right_leg_motor_2", "right_leg_motor_3",
                    "right_leg_motor_4", "right_leg_motor_5"


                    ]

            position = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            for joint, setpoint in self.get_setpoint(t).items():
                motor_index = name.index(joint)
                position[motor_index] = float(setpoint)

            if self.mirror:
                position_mirrored = copy.deepcopy(position)
                position_mirrored[0:2] = position[2:4]
                position_mirrored[2:4] = position[0:2]
                position_mirrored[4:10] = position[10:16]
                position_mirrored[10:16] = position[4:10]
                position = position_mirrored

            action = torch.tensor(position, dtype=torch.float, device=self.env.device)
            action = action - self.env.default_dof_pos[0]
            self.env.step(action)

            t = t + 0.00833
            time.sleep(0.00833)


class SoccerTrajectoryClass:
    def __init__(self, env, env_ids):
        self.trajectory_path = "/home/manx52/catkin_ws/src/rl-isaac-gym/resources/library/trajectories/trajectories"
        self.simulation = True
        self.trajectory_complete = True
        self.env = env
        self.env_ids = env_ids

    def run_trajectory(self, command: str):
        path = self.trajectory_path + "/" + "simulation_" + command + ".csv"

        if not os.path.exists(path):

            return

        print("Now publishing: ", command)
        trajectory = Trajectory(path, self.env, self.env_ids, False)
        trajectory.publish()
        print("Finished publishing:", command)


if __name__ == '__main__':
    trajectory_class = SoccerTrajectoryClass()

    trajectory_class.run_trajectory()
