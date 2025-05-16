"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)
        self.turn_counter = 0
        self.turn_direction = 0
        

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_tp2()
        #return self.control_tp4()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """

        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance

        pose = self.odometer_values()

        #self.tiny_slam.update_map(self.lidar(), pose)
        #self.occupancy_grid.display_cv(pose)

        command, rotate = reactive_obst_avoid(self.lidar())
        
         # if it's turning, keep turning for some frames
        if self.turn_counter > 0:
            self.turn_counter -= 1
            command = {"forward" : 0.0,
                       "rotation" : 0.5 * self.turn_direction}

        # if it detectates an obstacle and decides to turn
        elif rotate != 0:
            self.turn_direction = rotate
            self.turn_counter = 15  # turn for N frames
            command = {"forward" : 0.0,
                       "rotation" : 0.5 * self.turn_direction}

        # if no obstacle, it remains the same

        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        
        # Initializes the list of goals and the stage index
        if not hasattr(self, 'goals'):
            self.goals = [
                np.array([0, -200, 0]),
                np.array([-200, -400, 0]),
                np.array([-200, -200, 0]),
                np.array([-400, 0, 0])
            ]
            self.current_goal_index = 0

        current_goal = self.goals[self.current_goal_index]

        dist_to_goal = np.linalg.norm(pose[:2] - current_goal[:2])

        # If you got close, move on to the next goal, if there is one
        if dist_to_goal < 10.0 and self.current_goal_index < len(self.goals) - 1:
            self.current_goal_index += 1
            current_goal = self.goals[self.current_goal_index]

        self.occupancy_grid.display_cv(pose, current_goal)      
        command = potential_field_control(self.lidar(), pose, current_goal)

        return command

    def control_tp3(self):

        pose = self.odometer_values()

        self.tiny_slam.update_map(self.lidar(), pose)
        self.occupancy_grid.display_cv(pose)

        command = {"forward": 0.0,
                   "rotation": 0.0}
        return command

    def control_tp4(self):

        pose = self.odometer_values()
        best_score = self.tiny_slam.localise(self.lidar(), pose)
        # note_sur_20 = best_score*20/(self.occupancy_grid.max_grid_value*360)
        self.tiny_slam.update_map(self.lidar(), pose)
        #self.occupancy_grid.display_cv(pose)

        command = {"forward": 0.0,
                   "rotation": 0.0}
        return command
        