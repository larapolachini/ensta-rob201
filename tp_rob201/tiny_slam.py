""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        score = 0

        laser_dist = lidar.get_sensor_values()
        laser_ang = lidar.get_ray_angles()

        mask = (laser_dist > 0) & (laser_dist < lidar.max_range)
        laser_dist = laser_dist[mask]
        laser_ang = laser_ang[mask]


        theta_world = pose[2] + laser_ang
        x_world = (pose[0] + laser_dist*np.cos(pose[2] + theta_world))
        y_world = (pose[1] + laser_dist*np.sin(pose[2] + theta_world))

        idx_grid, idy_grid = self.grid.conv_world_to_map(x_world,y_world)

        valid_mask = (
        (idx_grid >= 0) & (idx_grid < self.grid.occupancy_map.shape[0]) &
        (idy_grid >= 0) & (idy_grid < self.grid.occupancy_map.shape[1])
        )

        idx_grid = idx_grid[valid_mask]
        idy_grid = idy_grid[valid_mask]

        idx_grid = idx_grid.astype(int)
        idy_grid = idy_grid.astype(int)

        logs = self.grid.occupancy_map[idx_grid, idy_grid]

        score = np.sum(logs)

        #print(score)

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4

        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        x_odom, y_odom, theta_odom = odom_pose
        x_ref, y_ref, theta_ref = odom_pose_ref

        # d = np.sqrt(x_odom**2 + y_odom**2)
        # a_o = np.arctan2(y_odom, x_odom)

        cos_theta_ref = np.cos(theta_ref)
        sin_theta_ref = np.sin(theta_ref)

        x_corrected = x_ref + x_odom * cos_theta_ref - y_odom * sin_theta_ref
        y_corrected = y_ref + x_odom * sin_theta_ref + y_odom * cos_theta_ref
        theta_corrected = theta_ref + theta_odom

        # Normalization of the angle
        theta_corrected = np.arctan2(np.sin(theta_corrected), np.cos(theta_corrected))

        corrected_pose = (x_corrected, y_corrected, theta_corrected)

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_odom_pose_ref = np.copy(self.odom_pose_ref)
        best_pose = self.get_corrected_pose(raw_odom_pose, self.odom_pose_ref) 
        best_score = self._score(lidar,best_pose)
        N = 30
        mu = 0 
        sigma = [0.05,0.05,0.01]

        #for _ in range(0, N):
        #    offset = np.random.normal(mu, sigma, 1)
        #    new_odom_pose = self.odom_pose_ref + offset
        #    new_pose = self.get_corrected_pose(raw_odom_pose, new_odom_pose) 
        #    new_score = self._score(lidar, new_pose) 
        #    if score < new_score:
        #        best_score = new_score  
                # self.odom_pose_ref += offset
        #    else: 
        #        best_score = score

        while mu < N:
            offset = np.random.normal(0, sigma, 3)  # (dx, dy, dtheta)
            new_odom_pose_ref = best_odom_pose_ref + offset
            new_pose = self.get_corrected_pose(raw_odom_pose, new_odom_pose_ref)
            new_score = self._score(lidar, new_pose)

            if new_score > best_score:
                best_score = new_score
                best_odom_pose_ref = new_odom_pose_ref
                mu = 0  # reset compteur
            else:
                mu += 1
        self.odom_pose_ref = best_odom_pose_ref
    
        print(best_score)

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        x_0 = pose[0]
        y_0 = pose[1]
        theta = pose[2]

        laser_dist = np.array(lidar.get_sensor_values())
        laser_ang = np.array(lidar.get_ray_angles())

        # Take 1 point out of 2
        laser_dist = laser_dist[::2]
        laser_ang = laser_ang[::2]


        # TODO for TP3

        # Converting to global coordenates
        lidar_x_i = x_0 + laser_dist*np.cos(theta + laser_ang)
        lidar_y_i = y_0 + laser_dist*np.sin(theta + laser_ang)

        # Updates the grid with free space
        for x_i, y_i in zip(lidar_x_i,lidar_y_i):
            self.grid.add_value_along_line(pose[0], pose[1], x_i, y_i, -1)

        # Update the obstacles
        self.grid.add_map_points(lidar_x_i, lidar_y_i, 5)

        # Limit for probability values
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -40, 40)

        
       


    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        points = []
        for i in range(3600):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
