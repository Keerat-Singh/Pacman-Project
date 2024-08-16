import random
import heapq
from . import Constants

class AStar:
    
    def __init__(self, board):
        self.board = board
        self.width = Constants.board_width
        self.height = Constants.board_height
        
    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x2, y2 = node[0] + dx, node[1] + dy
            if 0 <= x2 < self.width and 0 <= y2 < self.height and self.board[y2][x2] != 1:
                neighbors.append((x2, y2))
        return neighbors   

    def find_path(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: calculate_distance(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = calculate_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []
    

    def find_path_avoiding_pacman(self, start, goal, pacman_pos):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor == pacman_pos:
                    continue  # Avoid Pacman's position
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []


def can_move(self, new_x, new_y):
    # Check if the new position is within the screen boundaries
    if new_x < 0 or new_x >= len(self.board.map[0]) or new_y < 0 or new_y >= len(self.board.map):
        return False
    # Check if the new position is a valid cell-- '1' is wall
    return self.board.map[new_y][new_x] != 1

# Returns current position for the element, - for checking collision
def current_position(self):
    return (self.rect.x, self.rect.y)

def calculate_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def is_pacman_closer(self, pacman_pos, initial_pos):
    pacman_distance = calculate_distance(pacman_pos, initial_pos)
    ghost_distance = calculate_distance((self.rect.x, self.rect.y), initial_pos)
    return pacman_distance < ghost_distance

def run_away_from_pacman(self, pacman_pos):
    directions = Constants.DIRECTION[:]
    random.shuffle(directions)
    for direction in directions:
        ghost_x, ghost_y = self.rect.x, self.rect.y
        if direction == 'LEFT':
            ghost_x -= 1
        elif direction == 'RIGHT':
            ghost_x += 1
        elif direction == 'UP':
            ghost_y -= 1
        elif direction == 'DOWN':
            ghost_y += 1
        if can_move(self, ghost_x, ghost_y):
            if calculate_distance((ghost_x, ghost_y), pacman_pos) > calculate_distance((self.rect.x, self.rect.y), pacman_pos):
                self.rect.x = ghost_x
                self.rect.y = ghost_y
                break