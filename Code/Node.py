import numpy as np
from collections import deque

class Node:
    def __init__(self, x1, y1):
        self.edges = deque()  # all the nodes this node is connected to
        self.x = x1
        self.y = y1
        self.smallestDistToPoint = 10000000  # the distance of the smallest path from the start to this node
        self.degree = 0
        self.value = 0
        self.checked = False

    # Draw a little circle
    def show(self):
        print(f"Draw ellipse at ({self.x * 16 + 8}, {self.y * 16 + 8}) with diameter 10")  # replace with actual drawing code

    # Add all the nodes this node is adjacent to
    def add_edges(self, nodes, tiles):
        nodes_array = np.array([[node.x, node.y] for node in nodes])
        
        # Check for nodes on the same horizontal or vertical line
        horizontal = nodes_array[:, 1] == self.y
        vertical = nodes_array[:, 0] == self.x
        
        # Process horizontal edges
        if np.any(horizontal):
            horizontal_nodes = nodes_array[horizontal]
            for hx, hy in horizontal_nodes:
                if hx == self.x:
                    continue
                most_left = min(hx, self.x) + 1
                max_x = max(hx, self.x)
                edge = True
                for step in range(int(most_left), int(max_x)):
                    if tiles[int(self.y)][step].wall:
                        edge = False
                        break
                if edge:
                    self.edges.append(next(node for node in nodes if node.x == hx and node.y == hy))
        
        # Process vertical edges
        if np.any(vertical):
            vertical_nodes = nodes_array[vertical]
            for vx, vy in vertical_nodes:
                if vy == self.y:
                    continue
                most_up = min(vy, self.y) + 1
                max_y = max(vy, self.y)
                edge = True
                for step in range(int(most_up), int(max_y)):
                    if tiles[step][int(self.x)].wall:
                        edge = False
                        break
                if edge:
                    self.edges.append(next(node for node in nodes if node.x == vx and node.y == vy))

class Tile:
    def __init__(self, wall=False):
        self.wall = wall
