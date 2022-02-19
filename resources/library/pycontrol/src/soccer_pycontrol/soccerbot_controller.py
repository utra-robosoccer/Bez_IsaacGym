import os
from resources.library.geometry.src.soccer_geometry.transformation import Transformation
from time import sleep
from resources.library.pycontrol.src.soccer_pycontrol.soccerbot import Soccerbot

import torch
import time


class SoccerbotController:
    PYBULLET_STEP = 0.00833

    def __init__(self, env, env_ids):
        self.soccerbot = Soccerbot(Transformation(), env, env_ids)
        self.env = env

    def setGoal(self, goal: Transformation):
        self.soccerbot.setGoal(goal)

    def ready(self):
        self.soccerbot.ready()

    def wait(self, steps):
        for i in range(steps):
            time.sleep(SoccerbotController.PYBULLET_STEP)

    def run(self, stop_on_completed_trajectory=False):
        if self.soccerbot.robot_path.duration() == 0:
            return

        t = 0
        while t <= self.soccerbot.robot_path.duration():
            if self.soccerbot.current_step_time <= t <= self.soccerbot.robot_path.duration():
                self.soccerbot.stepPath(t, verbose=False)
                self.soccerbot.apply_imu_feedback(t, self.soccerbot.get_imu())

                action = torch.tensor(self.soccerbot.get_angles() , dtype=torch.float, device=self.env.device)
                action = action - self.env.default_dof_pos[0]

                self.env.step(action)

                self.soccerbot.current_step_time = self.soccerbot.current_step_time + self.soccerbot.robot_path.step_size

            t = t + SoccerbotController.PYBULLET_STEP
            sleep(SoccerbotController.PYBULLET_STEP)

    def updateGoal(self):
        pass
