#Murat Kacmaz - U254799
#Umut Caliskan - U254835

import random
import contest.util as util
from contest.captureAgents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearestPoint

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}

    def update_beliefs(self, game_state):
        for enemy in self.enemies:
            # Get current enemy position if visible
            enemy_pos = game_state.get_agent_position(enemy)

            if enemy_pos:
                # If we can see enemy, focus all belief on its position
                new_belief = util.Counter()
                new_belief[enemy_pos] = 1.0
            else:
                # Otherwise, update based on time transition model
                new_belief = util.Counter()

                # For each position the enemy could have been in
                for old_pos, prob in self.beliefs[enemy].items():
                    if prob > 0:
                        # Consider all possible moves from that position
                        for dx, dy in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:  # Include staying put
                            nx, ny = old_pos[0] + dx, old_pos[1] + dy

                            # Check if position is valid (not a wall and in bounds)
                            if 0 <= nx < self.width and 0 <= ny < self.height and not game_state.has_wall(nx, ny):
                                # Distribute probability to this new position
                                new_belief[(nx, ny)] += prob / 5  # Assume equal probability for each move

            if sum(new_belief.values()) > 0:
                new_belief.normalize()
                self.beliefs[enemy] = new_belief

    def heuristic(self, pos, goal):
        """Heuristic function for A* search - Manhattan distance"""
        return abs(int(pos[0]) - int(goal[0])) + abs(int(pos[1]) - int(goal[1]))

    def get_direction(self, current, next_pos):
        """Convert a position change to a game action"""
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]

        if dx == 1:
            return Directions.EAST
        elif dx == -1:
            return Directions.WEST
        elif dy == 1:
            return Directions.NORTH
        elif dy == -1:
            return Directions.SOUTH
        else:
            return Directions.STOP

    def initialize_beliefs(self, game_state):
        """Initialize belief distributions for enemy positions"""
        self.beliefs = {}
        for enemy in self.enemies:
            beliefs = util.Counter()
            # Start with uniform distribution over all valid positions
            for x in range(self.width):
                for y in range(self.height):
                    if not game_state.has_wall(x, y):
                        beliefs[(x, y)] = 1.0

            # If enemy is visible, put all belief on its position
            enemy_pos = game_state.get_agent_position(enemy)
            if enemy_pos:
                beliefs = util.Counter()
                beliefs[enemy_pos] = 1.0

            beliefs.normalize()
            self.beliefs[enemy] = beliefs

class OffensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.current_path = [] 
        self.current_goal = None 
        self.last_position = None 
        self.stuck_counter = 0 
        self.target_food = None  
        self.returning_home = False 
        self.beliefs = None  

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        if self.red:
            self.boundary = int(self.width / 2 - 1)
        else:
            self.boundary = int(self.width / 2)

        self.enemies = self.get_opponents(game_state)
        self.initialize_beliefs(game_state)
        self.distance_matrix = {}

    def is_ghost_nearby(self, game_state, pos, distance_threshold=2):
        """Check if there's a ghost near the given position"""
        for enemy in self.enemies:
            enemy_state = game_state.get_agent_state(enemy)
            if not enemy_state.is_pacman:  
                enemy_pos = game_state.get_agent_position(enemy)
                # If we can see the ghost, check direct distance
                if enemy_pos:
                    ghost_dist = self.get_maze_distance(pos, enemy_pos)
                    if ghost_dist <= distance_threshold:
                        return True, ghost_dist
                # If ghost not visible, check beliefs
                else:
                    for belief_pos, prob in self.beliefs[enemy].items():
                        if prob > 0.2:  
                            ghost_dist = self.get_maze_distance(pos, belief_pos)
                            if ghost_dist <= distance_threshold:
                                return True, ghost_dist

        return False, float('inf')

    def get_capsule_safety(self, game_state, my_pos):
        # Check if capsules are active (enemies scared)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghost_scared_times = [enemy.scared_timer for enemy in enemies if not enemy.is_pacman]

        if ghost_scared_times and max(ghost_scared_times) > 0:
            return True, max(ghost_scared_times)

        # Check if we can get to a capsule
        capsules = self.get_capsules(game_state)
        if capsules:
            min_cap_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            if min_cap_dist <= 3: 
                return True, 0

        return False, 0

    def a_star_search(self, game_state, start, goal, avoid_ghosts=True):
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        path_key = (start, goal, avoid_ghosts)
        if path_key in self.distance_matrix:
            return self.distance_matrix[path_key]

        open_set = util.PriorityQueue()
        open_set.push((start, []), self.heuristic(start, goal))

        closed_set = set()
        g_values = {start: 0}
        have_capsule, scared_timer = self.get_capsule_safety(game_state, start)

        while not open_set.isEmpty():
            (current, path) = open_set.pop()
            # If we've reached the goal, return the path
            if current == goal:
                # Cache result for future use
                self.distance_matrix[path_key] = path
                return path

            # Skip if we've already processed this node
            if current in closed_set:
                continue
            # Mark as visited
            closed_set.add(current)
            
            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x, next_y = int(x + dx), int(y + dy)
                next_pos = (next_x, next_y)

                # Check if position is valid (not a wall and in bounds)
                if (0 <= next_x < self.width and 0 <= next_y < self.height
                        and not game_state.has_wall(next_x, next_y)):
                    # Skip if already processed
                    if next_pos in closed_set:
                        continue
                    
                    dummy_g = g_values[current] + 1 

                    # Apply ghost avoidance if needed and we don't have capsule protection
                    ghost_penalty = 0
                    if avoid_ghosts and not have_capsule:
                        ghost_nearby, ghost_dist = self.is_ghost_nearby(game_state, next_pos, 3)
                        if ghost_nearby:
                            # Scale penalty based on distance
                            ghost_penalty = 100 * (4 - ghost_dist)

                    # Apply penalty to g value
                    dummy_g += ghost_penalty

                    # If we've found a better path to this node, update it
                    if next_pos not in g_values or dummy_g < g_values[next_pos]:
                        g_values[next_pos] = dummy_g
                        f_value = dummy_g + self.heuristic(next_pos, goal)
                        open_set.push((next_pos, path + [self.get_direction(current, next_pos)]), f_value)

        # No path found - return empty path
        return []

    def find_best_food_cluster(self, game_state, my_pos, max_cluster_size=5):
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return None

        clusters = []
        min_dist = float('inf')
        closest_food = None

        for food in food_list:
            dist = self.get_maze_distance(my_pos, food)
            if dist < min_dist:
                min_dist = dist
                closest_food = food

        if closest_food:
            clusters.append([closest_food])

        while len(food_list) > len([f for c in clusters for f in c]):
            for cluster in clusters:
                if len(cluster) >= max_cluster_size:
                    continue

                # Find closest food to this cluster
                cluster_center = cluster[0] 
                min_dist = float('inf')
                closest_food = None

                for food in food_list:
                    if food in [f for c in clusters for f in c]:
                        continue  

                    dist = self.get_maze_distance(cluster_center, food)
                    if dist < min_dist and dist <= 2: 
                        min_dist = dist
                        closest_food = food

                if closest_food:
                    cluster.append(closest_food)

            # If no more food can be added to existing clusters, create a new one
            remaining_food = [f for f in food_list if f not in [f for c in clusters for f in c]]
            if remaining_food:
                clusters.append([remaining_food[0]])
            else:
                break

        # Evaluate clusters based on value and distance
        best_cluster = None
        best_value = float('-inf')

        for cluster in clusters:
            cluster_size = len(cluster)
            cluster_center = cluster[0]
            distance = self.get_maze_distance(my_pos, cluster_center)

            # Compute ghost risk for this cluster
            ghost_risk = 0
            for food in cluster:
                ghost_nearby, ghost_dist = self.is_ghost_nearby(game_state, food, 3)
                if ghost_nearby:
                    ghost_risk += (4 - ghost_dist) / len(cluster)

            value = cluster_size / (distance + 1) - ghost_risk

            if value > best_value:
                best_value = value
                best_cluster = cluster

        return best_cluster[0] if best_cluster else closest_food

    def choose_action(self, game_state):
        # Update beliefs about enemies
        self.update_beliefs(game_state)

        my_pos = game_state.get_agent_state(self.index).get_position()
        if self.last_position == my_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_position = my_pos

        if self.stuck_counter >= 3:
            self.current_path = []
            self.target_food = None

        carrying = game_state.get_agent_state(self.index).num_carrying
        food_left = len(self.get_food(game_state).as_list())
        ghost_nearby, ghost_dist = self.is_ghost_nearby(game_state, my_pos, 5)
        have_capsule, scared_timer = self.get_capsule_safety(game_state, my_pos)
        return_home_conditions = (
                (carrying > 5 or food_left <= 2) or
                (ghost_nearby and carrying > 0 and ghost_dist < 3 and not have_capsule)
        )

        if return_home_conditions and not self.returning_home:
            self.returning_home = True
            self.current_path = [] 

        if self.returning_home and carrying == 0 and (
                (self.red and my_pos[0] <= self.boundary) or
                (not self.red and my_pos[0] >= self.boundary)
        ):
            self.returning_home = False
            self.current_path = [] 

        if not self.current_path:
            # If returning home, find path to nearest home position
            if self.returning_home:
                # Find nearest position on our side
                home_positions = []
                if self.red:
                    home_positions = [(self.boundary, y) for y in range(self.height)
                                      if not game_state.has_wall(self.boundary, y)]
                else:
                    home_positions = [(self.boundary + 1, y) for y in range(self.height)
                                      if not game_state.has_wall(self.boundary + 1, y)]

                if home_positions:
                    min_dist = float('inf')
                    closest_home = None
                    for pos in home_positions:
                        dist = self.get_maze_distance(my_pos, pos)
                        if dist < min_dist:
                            min_dist = dist
                            closest_home = pos

                    if closest_home:
                        # Plan path to home, avoiding ghosts if carrying food
                        self.current_goal = closest_home
                        self.current_path = self.a_star_search(game_state, my_pos, closest_home, avoid_ghosts=True)

            else:
                food_list = self.get_food(game_state).as_list()

                if food_list:
                    # If ghosts are nearby, try to get a capsule first
                    capsules = self.get_capsules(game_state)
                    if ghost_nearby and capsules and not have_capsule:
                        min_dist = float('inf')
                        closest_capsule = None
                        for capsule in capsules:
                            dist = self.get_maze_distance(my_pos, capsule)
                            if dist < min_dist:
                                min_dist = dist
                                closest_capsule = capsule

                        if closest_capsule and min_dist < 10:  # Only go for capsule if it's reasonably close
                            self.current_goal = closest_capsule
                            self.current_path = self.a_star_search(game_state, my_pos, closest_capsule,
                                                                   avoid_ghosts=True)

                    else:
                        target = self.find_best_food_cluster(game_state, my_pos)
                        if target:
                            self.target_food = target
                            self.current_goal = target
                            avoid = not have_capsule
                            self.current_path = self.a_star_search(game_state, my_pos, target, avoid_ghosts=avoid)

        # If we have a path, follow it
        if self.current_path:
            next_action = self.current_path[0]
            self.current_path = self.current_path[1:]
            if next_action in game_state.get_legal_actions(self.index):
                return next_action

        return super().choose_action(game_state)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)
        my_pos = successor.get_agent_state(self.index).get_position()

        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        scared_ghosts = [a for a in ghosts if a.scared_timer > 0]

        if ghosts:
            dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            features['ghost_distance'] = min(dists)

            # Check if all ghosts are scared
            if len(scared_ghosts) == len(ghosts):
                features['all_ghosts_scared'] = 1
                # If ghosts are scared, we want to get closer to them
                features['ghost_distance'] = -features['ghost_distance']

            if min(dists) <= 1 and len(scared_ghosts) != len(ghosts):
                features['ghost_collision'] = 1
                
        capsules = self.get_capsules(successor)
        if len(capsules) > 0:
            min_cap_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            features['capsule_distance'] = min_cap_dist

            if ghosts and min(dists) < 5 and len(scared_ghosts) != len(ghosts):
                features['capsule_priority'] = 1

        # Home return with food
        carrying = successor.get_agent_state(self.index).num_carrying
        if carrying > 0:
            # Find closest position on our side
            if self.red:
                home_positions = [(self.boundary, y) for y in range(self.height) if
                                  not successor.has_wall(self.boundary, y)]
            else:
                home_positions = [(self.boundary + 1, y) for y in range(self.height) if
                                  not successor.has_wall(self.boundary + 1, y)]

            if home_positions:
                min_home_dist = min([self.get_maze_distance(my_pos, pos) for pos in home_positions])
                features['return_home'] = min_home_dist * carrying

                # If ghosts are nearby and we're carrying food, strongly prioritize returning
                if ghosts and min(dists) < 5 and len(scared_ghosts) != len(ghosts):
                    features['return_urgency'] = min_home_dist * carrying * 2

        return features

    def get_weights(self, game_state, action):
        carrying = game_state.get_agent_state(self.index).num_carrying
        weights = {
            'successor_score': 100,
            'distance_to_food': -2,
            'ghost_distance': 50,  # Positive because closer is worse (unless ghosts are scared)
            'ghost_collision': -1000,  # Very high penalty for ghost collision
            'capsule_distance': -10,
            'capsule_priority': -20,
            'return_home': -5,
            'return_urgency': -20,
            'all_ghosts_scared': 100,  # Bonus for all ghosts being scared
        }

        if carrying > 0:
            factor = min(5, carrying) / 2  # Scale by amount of food, but cap at 2.5x
            weights['return_home'] = -20 * factor
            weights['distance_to_food'] = -1  # Less interested in food
            weights['ghost_distance'] = 100 * factor  # More afraid of ghosts
            weights['ghost_collision'] = -2000 * factor  # Even more afraid of collision

        return weights

class DefensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.current_path = [] 
        self.current_goal = None 
        self.last_position = None 
        self.stuck_counter = 0
        self.patrol_positions = []
        self.patrol_index = 0 
        self.beliefs = None 
        self.distance_matrix = {}

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        if self.red:
            self.boundary = int(self.width / 2 - 1)
            self.defense_area = list(range(self.boundary + 1))
        else:
            self.boundary = int(self.width / 2)
            self.defense_area = list(range(self.boundary, self.width))

        self.generate_patrol_positions(game_state)
        self.enemies = self.get_opponents(game_state)
        self.initialize_beliefs(game_state)

    def generate_patrol_positions(self, game_state):
        """Generate strategic positions to patrol on our side"""
        patrol_positions = []

        # First priority: add entry points to our territory (boundary positions)
        boundary_positions = []
        for y in range(self.height):
            if self.red:
                # For red team, patrol the right boundary of our territory
                if not game_state.has_wall(self.boundary, y):
                    boundary_positions.append((self.boundary, y))
            else:
                # For blue team, patrol the left boundary of our territory
                if not game_state.has_wall(self.boundary, y):
                    boundary_positions.append((self.boundary, y))

        # Sort boundary positions by their tactical value (middle positions first)
        mid_y = self.height / 2
        boundary_positions.sort(key=lambda pos: abs(pos[1] - mid_y))
        patrol_positions.extend(boundary_positions)

        # Second priority: add food locations on our side
        our_food = self.get_food_you_are_defending(game_state).as_list()

        # Sort food by distance from boundary
        food_by_importance = []
        for food_pos in our_food:
            min_dist = min([self.get_maze_distance(food_pos, boundary) for boundary in boundary_positions])
            food_by_importance.append((food_pos, min_dist))

        # Add closest food to boundary first (most vulnerable)
        food_by_importance.sort(key=lambda x: x[1])
        patrol_positions.extend([food[0] for food in food_by_importance])

        # Third priority: add capsule locations
        our_capsules = self.get_capsules_you_are_defending(game_state)
        patrol_positions.extend(our_capsules)

        # Filter to ensure positions are on our side
        self.patrol_positions = [pos for pos in patrol_positions if
                                 (self.red and pos[0] <= self.boundary) or
                                 (not self.red and pos[0] >= self.boundary)]

        # Ensure we have at least some patrol positions
        if not self.patrol_positions:
            # Add some fallback positions on our side
            for x in self.defense_area:
                for y in range(self.height):
                    if not game_state.has_wall(x, y):
                        self.patrol_positions.append((x, y))
                        if len(self.patrol_positions) >= 5:  # Limit to 5 positions
                            break

    def a_star_search(self, game_state, start, goal, avoid_ghosts=False):
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        path_key = (start, goal, avoid_ghosts)
        if path_key in self.distance_matrix:
            return self.distance_matrix[path_key]
        open_set = util.PriorityQueue()
        open_set.push((start, []), self.heuristic(start, goal))

        closed_set = set()
        g_values = {start: 0}

        while not open_set.isEmpty():
            (current, path) = open_set.pop()
            if current == goal:
                # Cache result for future use
                self.distance_matrix[path_key] = path
                return path

            if current in closed_set:
                continue

            closed_set.add(current)
            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_x, next_y = int(x + dx), int(y + dy)
                next_pos = (next_x, next_y)

                if (0 <= next_x < self.width and 0 <= next_y < self.height
                        and not game_state.has_wall(next_x, next_y)):
                    if next_pos in closed_set:
                        continue

                    in_our_territory = (self.red and next_x <= self.boundary) or \
                                       (not self.red and next_x >= self.boundary)

                    dummy_g = g_values[current] + 1 
                    if not in_our_territory and not avoid_ghosts:
                        dummy_g += 5

                    if avoid_ghosts:  
                        invader_bonus = 0
                        for enemy in self.enemies:
                            for pos, prob in self.beliefs[enemy].items():
                                invader_pos = (self.red and pos[0] <= self.boundary) or \
                                              (not self.red and pos[0] >= self.boundary)

                                if invader_pos and prob > 0.1:
                                    dist = self.get_maze_distance(next_pos, pos)
                                    if dist < 5: 
                                        # Closer positions and higher probabilities get bigger bonus
                                        invader_bonus += 5.0 * prob / (dist + 1)

                        # Reduce g value to prioritize paths toward invaders
                        dummy_g -= invader_bonus

                    # If we've found a better path to this node, update it
                    if next_pos not in g_values or dummy_g < g_values[next_pos]:
                        g_values[next_pos] = dummy_g
                        f_value = dummy_g + self.heuristic(next_pos, goal)
                        open_set.push((next_pos, path + [self.get_direction(current, next_pos)]), f_value)

        # No path found - return empty path
        return []

    def find_closest_invader(self, game_state, my_pos):
        invaders = []
        for enemy in self.enemies:
            enemy_state = game_state.get_agent_state(enemy)
            enemy_pos = game_state.get_agent_position(enemy)

            if enemy_state.is_pacman and enemy_pos is not None:
                dist = self.get_maze_distance(my_pos, enemy_pos)
                invaders.append((enemy_pos, dist, 1.0))  # position, distance, certainty

        if invaders:
            invaders.sort(key=lambda x: x[1])  
            return invaders[0][0], invaders[0][2] 

        potential_invaders = []
        for enemy in self.enemies:
            # For each position with some probability
            for pos, prob in self.beliefs[enemy].items():
                # Check if in our territory (potential invader)
                is_invader = (self.red and pos[0] <= self.boundary) or \
                             (not self.red and pos[0] >= self.boundary)

                if is_invader and prob > 0.1:
                    dist = self.get_maze_distance(my_pos, pos)
                    potential_invaders.append((pos, dist, prob))

        # Return the most likely invader position
        if potential_invaders:
            # Sort by a combination of certainty and distance
            potential_invaders.sort(key=lambda x: x[1] / x[2])  # distance / certainty
            return potential_invaders[0][0], potential_invaders[0][2]

        return None, 0

    def get_next_patrol_point(self, game_state, my_pos):
        if not self.patrol_positions:
            return None

        defending_food = self.get_food_you_are_defending(game_state)
        missing_food = set(self.patrol_positions) - set(defending_food.as_list())
        missing_food = [pos for pos in missing_food if (self.red and pos[0] <= self.boundary) or
                        (not self.red and pos[0] >= self.boundary)]

        if missing_food:
            closest_pos = None
            min_dist = float('inf')

            for pos in self.patrol_positions:
                for missing in missing_food:
                    dist = self.get_maze_distance(pos, missing)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pos = pos

            if closest_pos:
                return closest_pos

        weights = []

        for pos in self.patrol_positions:
            weight = 3.0 if pos[0] == self.boundary else 1.0
            mid_y = self.height / 2
            y_dist = abs(pos[1] - mid_y)
            weight *= (self.height - y_dist) / self.height
            dist = self.get_maze_distance(my_pos, pos)
            if dist > 0:
                weight *= 5.0 / dist

            weights.append(weight)

        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

            r = random.random()
            cumulative = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    return self.patrol_positions[i]

        self.patrol_index = (self.patrol_index + 1) % len(self.patrol_positions)
        return self.patrol_positions[self.patrol_index]

    def choose_action(self, game_state):
        self.update_beliefs(game_state)
    
        my_pos = game_state.get_agent_state(self.index).get_position()
        my_state = game_state.get_agent_state(self.index)
        if self.last_position == my_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_position = my_pos
        if self.stuck_counter >= 3:
            self.current_path = []
    
        # If we're a pacman, try to return to our side quickly
        if my_state.is_pacman:
            # Find the fastest way back to our territory
            home_pos = []
            if self.red:
                home_pos = [(self.boundary, y) for y in range(self.height)
                            if not game_state.has_wall(self.boundary, y)]
            else:
                home_pos = [(self.boundary + 1, y) for y in range(self.height)
                            if not game_state.has_wall(self.boundary + 1, y)]
    
            if home_pos:
                closest_home = min(home_pos, key=lambda x: self.get_maze_distance(my_pos, x))
                path = self.a_star_search(game_state, my_pos, closest_home, False)
                if path:
                    return path[0]
    
        # First priority: check for visible invaders
        invaders = []
        for enemy in self.enemies:
            enemy_state = game_state.get_agent_state(enemy)
            enemy_pos = game_state.get_agent_position(enemy)
    
            if enemy_state.is_pacman and enemy_pos is not None:
                invaders.append((enemy, enemy_pos))
    
        if invaders:
            # Go after the closest invader
            closest_invader = min(invaders, key=lambda x: self.get_maze_distance(my_pos, x[1]))
            invader_pos = closest_invader[1]
    
            # If we're very close to the invader, just move directly toward it
            dist_to_invader = self.get_maze_distance(my_pos, invader_pos)
            if dist_to_invader <= 1:
                action = self.get_direction(my_pos, invader_pos)
                if action in game_state.get_legal_actions(self.index):
                    return action
    
            # Plan path to invader
            self.current_path = self.a_star_search(game_state, my_pos, invader_pos, True)
            if self.current_path:
                return self.current_path[0]
    
        # Second priority: check for likely invaders based on beliefs
        invader_pos, certainty = self.find_closest_invader(game_state, my_pos)
        if invader_pos and certainty >= 0.3:  # Only chase if somewhat certain
            # Plan path to likely invader
            self.current_path = self.a_star_search(game_state, my_pos, invader_pos, True)
            if self.current_path:
                return self.current_path[0]
    
        # Third priority: patrol strategic positions
        if not self.current_path:
            patrol_pos = self.get_next_patrol_point(game_state, my_pos)
            if patrol_pos:
                self.current_path = self.a_star_search(game_state, my_pos, patrol_pos, False)
    
        # If we have a path, follow it
        if self.current_path:
            next_action = self.current_path[0]
            self.current_path = self.current_path[1:]
    
            # Safety check: make sure the action is legal
            if next_action in game_state.get_legal_actions(self.index):
                return next_action
    
        # Fallback: choose a random legal action
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        return random.choice(legal_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if len(invaders) == 0:
            invader_pos, certainty = self.find_closest_invader(game_state, my_pos)
            if invader_pos and certainty > 0.1:
                features['believed_invader_distance'] = self.get_maze_distance(my_pos, invader_pos)
                features['believed_invader_certainty'] = certainty

        boundary_dist = abs(my_pos[0] - self.boundary)
        features['boundary_distance'] = boundary_dist

        if self.patrol_positions:
            patrol_dist = min([self.get_maze_distance(my_pos, p) for p in self.patrol_positions])
            features['patrol_distance'] = patrol_dist

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
            
        features['random'] = random.random() * 0.01

        return features

    def get_weights(self, game_state, action):
        # Dynamic weights based on game state
        weights = {
            'num_invaders': -1000,
            'invader_distance': -20,
            'believed_invader_distance': -10,
            'believed_invader_certainty': 100,
            'on_defense': 100,
            'boundary_distance': -5,
            'patrol_distance': -1,
            'stop': -100,
            'reverse': -2,
            'random': 1
        }

        # If there are invaders, focus on catching them
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        if invaders:
            weights['invader_distance'] = -40
            weights['boundary_distance'] = 0  # Don't care about boundary when actively chasing

        return weights
