from collections import deque
import math

class PVector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Path:
    def __init__(self):
        self.path = deque()  # a deque of nodes
        self.distance = 0  # length of path
        self.distToFinish = 0  # the distance between the final node and the path's goal
        self.velAtLast = None  # the direction the ghost is going at the last point on the path

    # Adds a node to the end of the path
    def add_to_tail(self, n, end_node):
        if self.path:  # if path is not empty
            last_node = self.path[-1]
            self.distance += math.dist((last_node.x, last_node.y), (n.x, n.y))  # add the distance from the current last element in the path to the new node to the overall distance

        self.path.append(n)  # add the node
        self.distToFinish = math.dist((self.path[-1].x, self.path[-1].y), (end_node.x, end_node.y))  # recalculate the distance to the finish

    # Return a clone of this
    def clone(self):
        temp = Path()
        temp.path = self.path.copy()
        temp.distance = self.distance
        temp.distToFinish = self.distToFinish
        temp.velAtLast = PVector(self.velAtLast.x, self.velAtLast.y)
        return temp

    # Removes all nodes in the path
    def clear(self):
        self.distance = 0
        self.distToFinish = 0
        self.path.clear()

    # Draw lines representing the path
    def show(self):
        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i].x * 16 + 8, self.path[i].y * 16 + 8
            x2, y2 = self.path[i + 1].x * 16 + 8, self.path[i + 1].y * 16 + 8
            print(f"Draw line from ({x1}, {y1}) to ({x2}, {y2})")  # replace with actual drawing code

        if self.path:
            x, y = self.path[-1].x * 16 + 8, self.path[-1].y * 16 + 8
            print(f"Draw ellipse at ({x}, {y})")  # replace with actual drawing code
