"""
Planner class
Implementation of A*
"""

import numpy as np

from occupancy_grid import OccupancyGrid
import heapq

class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid


        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        # TODO for TP5

        start_cell = tuple(self.grid.conv_world_to_map(start[0], start[1]))
        goal_cell = tuple(self.grid.conv_world_to_map(goal[0], goal[1]))

        # A* structures
        open_set = []
        heapq.heappush(open_set, (0, start_cell))  # (fScore, node)
        came_from = {}

        g_score = {start_cell: 0}
        f_score = {start_cell: self.heuristic(start_cell, goal_cell)}

        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_cell:
                #Reconstruct path
                path_cells = [current]

                while current in came_from:
                    current = came_from[current]
                    path_cells.insert(0, current)



                # Convertir le chemin grille -> monde
                path_world = []
                for cell in path_cells:
                    x, y = self.grid.conv_map_to_world(cell[0], cell[1])
                    path_world.append(np.array([x, y, 0]))   #theta = 0
                return path_world
            
            visited.add(current)

            for neighbor in self.get_neighbors(current):

                x,y = neighbor

                if not (0 <= x < self.grid.occupancy_map.shape[0] and 
                        0 <= y < self.grid.occupancy_map.shape[1]):
                    continue

                # BUFFER DE SEGURANÇA: verifica arredores do vizinho
                neighborhood = self.grid.occupancy_map[max(0, x-1):min(x+2, self.grid.occupancy_map.shape[0]),
                                           max(0, y-1):min(y+2, self.grid.occupancy_map.shape[1])]
                if np.any(neighborhood > 15):  # pode ajustar esse threshold
                    continue

                #Ne pas traverser les murs (valeurs > 0 dans la carte de probas)
                #if self.grid.occupancy_map[neighbor[0], neighbor[1]] > 0.3:
                 #   continue

                tentative_g = g_score[current] + self.heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_cell)

                    if neighbor not in visited:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

        #path = [start, goal]  # list of poses
        #return path

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal

    def get_neighbors(self, current_cell):
        """
        Retourne les 8 voisins (en coordonnées map) d'une cellule donnée,
        en tenant compte des bords de la carte.
        """
        x, y = current_cell
        neighbors = []

        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid.occupancy_map.shape[0] and 0 <= ny < self.grid.occupancy_map.shape[1]:
                neighbors.append((nx, ny))

        return neighbors
    
    def heuristic(self, cell1, cell2):
        """
        Distance euclidienne entre deux cellules
        """
        x1, y1 = cell1
        x2, y2 = cell2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)



           
        