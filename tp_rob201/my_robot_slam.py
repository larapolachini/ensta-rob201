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
        return self.control_tp5()
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
        #self.occupancy_grid.display_cv(pose)
        
        # Show map only every 10 iterations
        if not hasattr(self, "display_counter"):
            self.display_counter = 0

        if self.display_counter % 10 == 0:
            self.occupancy_grid.display_cv(pose)

        self.display_counter += 1
        

        # Use keyboard or random movement
        command = {"forward": 0.08,
                   "rotation": 0.05}
        return command

    def control_tp4(self):
        pose = self.odometer_values()

        # Etape SLAM : localisation avec correction de pose
        score = self.tiny_slam.localise(self.lidar(), pose)
        corrected_pose = self.tiny_slam.get_corrected_pose(pose)

        # Initialiser les objectifs si pas encore fait
        if not hasattr(self, 'goals'):
            self.goals = [
                np.array([0, -200, 0]), 
                np.array([-200, -400, 0]),
                np.array([-400, -500, 0]),
                np.array([-500, -400, 0]),
                np.array([-400, -500, 0]),
                np.array([-200, -200, 0]),
                np.array([-400, 0, 0]),
                np.array([-150, 0, 0]), #canto direito
                np.array([-500, 0,0]),
                np.array([-500, -250, 0]),
                np.array([-500,0,0]),
                np.array([-800, -100,0]),
                np.array([-700, -200, 0]),
                np.array([-700, -400, 0]),
                np.array([-800, -100, 0]),
                np.array([-800, -300, 0]),
                np.array([-900, -300, 0]),
                np.array([-900, -50, 0]),
                np.array([-900, -300, 0]),
                np.array([-800, -300, 0]),
                np.array([-800, -400, 0]),
                np.array([-900, -400, 0])

            ]
            self.current_goal_index = 0

        current_goal = self.goals[self.current_goal_index]
        dist_to_goal = np.linalg.norm(corrected_pose[:2] - current_goal[:2])

        if dist_to_goal < 10.0 and self.current_goal_index < len(self.goals) - 1:
            self.current_goal_index += 1
            current_goal = self.goals[self.current_goal_index]

    # Mise à jour de la carte si localisation est suffisamment bonne
        if -score > 50 or self.counter < 50:
            self.tiny_slam.update_map(self.lidar(), corrected_pose)

    # Affichage de la carte
        if self.counter % 10 == 0:
            self.occupancy_grid.display_cv(corrected_pose, current_goal)

    # Commande de mouvement : navigation vers l'objectif corrigé
        command = potential_field_control(self.lidar(), corrected_pose, current_goal)

        return command
        


    def control_tp5(self):
        pose = self.odometer_values()
        corrected_pose = self.tiny_slam.get_corrected_pose(pose)

        # Inicializa atributos na primeira chamada
        if not hasattr(self, "counter"):
            self.counter = 0
        if not hasattr(self, "planner"):
            from planner import Planner
            self.planner = Planner(self.occupancy_grid)
        if not hasattr(self, "path_planned"):
            self.path_planned = False
        if not hasattr(self, "path_index"):
            self.path_index = 0

        # Fase 1: Exploração com SLAM por 200 iterações
        if self.counter < 9000:           
            self.tiny_slam.localise(self.lidar(), pose)
            self.tiny_slam.update_map(self.lidar(), corrected_pose)
            
            if not hasattr(self, 'goals'):
                self.goals = [
                    np.array([0, -200, 0]), 
                    np.array([-200, -400, 0]),
                    np.array([-400, -500, 0]),
                    np.array([-500, -400, 0]),
                    np.array([-400, -500, 0]),
                    np.array([-200, -200, 0]),
                    np.array([-400, 0, 0]),
                    np.array([-150, 0, 0]), #canto direito
                    np.array([-500, 0,0]),
                    np.array([-500, -250, 0]),
                    np.array([-500,0,0]),
                    np.array([-800, -100,0]),
                    np.array([-700, -200, 0]),
                    np.array([-700, -400, 0]),
                    np.array([-800, -100, 0]),
                    np.array([-800, -300, 0]),
                    np.array([-900, -300, 0]),
                    np.array([-900, -50, 0]),
                    np.array([-900, -300, 0]),
                    np.array([-800, -300, 0]),
                    np.array([-800, -400, 0]),
                    np.array([-900, -400, 0])
                ]
                self.current_goal_index = 0

            current_goal = self.goals[self.current_goal_index]
            dist_to_goal = np.linalg.norm(corrected_pose[:2] - current_goal[:2])

            if dist_to_goal < 10.0 and self.current_goal_index < len(self.goals) - 1:
                self.current_goal_index += 1
                current_goal = self.goals[self.current_goal_index]

            self.occupancy_grid.display_cv(corrected_pose, current_goal)
            command = potential_field_control(self.lidar(), corrected_pose, current_goal)

        # Fase 2: Planejamento do caminho de volta
        elif not self.path_planned:
            start = corrected_pose
            goal = np.array([0, 0, 0])
            self.path = self.planner.plan(start, goal)
            self.path_planned = True
            self.path_index = 0
            command = {"forward": 0.0, "rotation": 0.0}

        # Fase 3: Seguimento do caminho até (0, 0, 0)
        else:
            if self.path_index < len(self.path):
                target = self.path[self.path_index]
                dist = np.linalg.norm(corrected_pose[:2] - target[:2])

            # Avança para o próximo ponto se estiver próximo
                if dist < 20.0:
                    self.path_index += 1
                    if self.path_index < len(self.path):
                        target = self.path[self.path_index]

                if self.path_index < len(self.path):
                    command = potential_field_control(self.lidar(), corrected_pose, target)
                else:
                    command = {"forward": 0.0, "rotation": 0.0}
            else:
                command = {"forward": 0.0, "rotation": 0.0}

        # Visualização do mapa com trajetória, se disponível
        if hasattr(self, "path"):
            if isinstance(self.path, list):
                self.path = np.array(self.path)

            smoothed_path = self.planner.interpolate_path(self.path)
                #self.occupancy_grid.display_cv(corrected_pose, np.array([0, 0, 0]), self.path)
        #else:
         #   self.occupancy_grid.display_cv(corrected_pose)

        self.counter += 1
        return command


