""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """


    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    speed = 0.0
    rotation_speed = 0.0

    #180 is the front of the car

    front_sector = laser_dist[170:190]  
    front_min = np.min(front_sector)  
    rotate = 0

    if front_min < 50:
        left_dist = laser_dist[0]   # get laser left
        right_dist = laser_dist[360]  # get laser right
        if left_dist > right_dist:
            rotate = -1
        else:
            rotate = 1
        speed = 0.0
        

    else:
        speed = 0.3
        rotation_speed = 0

    rotate_ind = 0.5
    command = {"forward": speed,
               "rotation": rotation_speed*rotate_ind}
    
    return command, rotate

def segment_lidar(distances, angles, seuil=10.0):
    clusters = []
    current_cluster = []

    for i in range(len(distances)):
        if not current_cluster:
            current_cluster.append((distances[i], angles[i]))
        else:
            prev_d, _ = current_cluster[-1]
            if abs(distances[i] - prev_d) < seuil:
                current_cluster.append((distances[i], angles[i]))
            else:
                clusters.append(current_cluster)
                current_cluster = [(distances[i], angles[i])]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters


def grad_attr(current_pose, goal_pose, K=0.2, d_switch=1.0):
    vec_goal = goal_pose[:2] - current_pose[:2]
    d_q = np.linalg.norm(vec_goal)

    if d_q < 1e-5:
        grad_attr = np.zeros(2)
    elif d_q < d_switch:
        # Potentiel quadratique
        grad_attr = K * vec_goal
    else:
        # Potentiel linéaire (classique)
        grad_attr = (K / d_q) * vec_goal

    return d_q, grad_attr

def grad_rep(lidar, current_pose, d_s = 30, K_o = 50):
    laser_dist = lidar.get_sensor_values()
    laser_ang = lidar.get_ray_angles()

    valid = (laser_dist > 5.0) & (laser_dist < 300.0)
    laser_dist = laser_dist[valid]
    laser_ang = laser_ang[valid]

    grad_rep = np.zeros(2)
    x, y, theta = current_pose
    q = np.array([x, y])

    clusters = segment_lidar(laser_dist, laser_ang, seuil=10.0)


    for d, a in zip(laser_dist, laser_ang):
        if d < d_s:
            obs_x = x + d * np.cos(theta + a)
            obs_y = y + d * np.sin(theta + a)
            q_obs = np.array([obs_x, obs_y])
            q = np.array([x, y])
            delta = q - q_obs
            dist = np.linalg.norm(delta)

            if dist > 1e-3:
                repulsion = K_o * (1.0/dist - 1.0/d_s) * (1.0 / dist**3) * delta
                grad_rep += repulsion

    return grad_rep


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2
    speed = 0.05
    speed_turn = 0

    max_speed = 0.10    
    max_turn = 0.5      

     # Attractif
    d_q, grad = grad_attr(current_pose, goal_pose)
    #print(d_q)    

    if d_q < 5.0:
        speed = 0
        speed_turn = 0

    else:
        grad_repulsive = grad_rep(lidar, current_pose, d_s=30) 
        grad_total = grad + grad_repulsive

        desired_angle = np.arctan2(grad_total[1], grad_total[0])
        angle_dif = desired_angle - current_pose[2]
        angle_dif = np.arctan2(np.sin(angle_dif), np.cos(angle_dif))

        speed = max_speed * np.clip(d_q, 0.0, 1.0) 
        speed_turn = np.clip(2.0 * angle_dif, -max_turn, max_turn)

        # Initial direction
        if abs(angle_dif) > np.pi/4:
            speed = 0.0

    command = {"forward": speed,
               "rotation": speed_turn}

    return command