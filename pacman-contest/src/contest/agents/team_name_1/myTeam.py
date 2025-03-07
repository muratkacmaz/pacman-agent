# baselineTeam.py
# ---------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import random
import util
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveHeuristicAgent', 
                second='DefensiveHeuristicAgent', 
                num_training=0):
    """Creates a team of two agents."""
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """Base class for reflex agents that maximize scores."""
    
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """Picks among highest Q-value actions or returns to start if food is low."""
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return Directions.STOP

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = Directions.STOP
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos)
                if dist < best_dist:
                    best_dist = dist
                    best_action = action
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """Finds the next grid position successor."""
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        return successor.generate_successor(self.index, action) if pos != nearestPoint(pos) else successor

    def evaluate(self, game_state, action):
        """Computes linear combination of features and weights."""
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

class OffensiveHeuristicAgent(ReflexCaptureAgent):
    """Reflex agent that seeks food and power dots, escapes ghosts effectively when close, and chases when powered."""
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_list = self.get_food(successor).as_list()
        
        # Score based on remaining food
        features['successor_score'] = -len(food_list)

        # Distance to nearest regular food
        if food_list:
            min_distance = min(self.get_maze_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_distance

        # Distance to nearest power dot (capsule)
        capsules = self.get_capsules(successor)  # Power dots are capsules in the game
        if capsules:
            min_capsule_dist = min(self.get_maze_distance(my_pos, capsule) for capsule in capsules)
            features['distance_to_power_dot'] = min_capsule_dist

        # Ghost and enemy handling
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position()]
        pacmen = [a for a in enemies if a.is_pacman and a.get_position()]

        # Power pellet status (big dot eaten)
        is_powered = my_state.scared_timer > 0  # True when Pacman has eaten a power pellet

        # Calculate ghost distances if ghosts exist
        ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts] if ghosts else []

        if is_powered and pacmen:  # Chase invaders when powered
            invader_dists = [self.get_maze_distance(my_pos, p.get_position()) for p in pacmen]
            if invader_dists:
                features['distance_to_invader'] = min(invader_dists)
        elif ghost_dists:  # Escape ghosts when not powered
            min_ghost_dist = min(ghost_dists)
            if min_ghost_dist <= 3:  # Prioritize escape if ghost is within 3 moves (closer than 4)
                features['escape_priority'] = min_ghost_dist
            elif min_ghost_dist <= 5:  # Moderate avoidance if ghost is nearby but beyond 3 moves
                features['distance_to_ghost'] = min_ghost_dist
            else:
                features['distance_to_ghost'] = 0  # Ignore distant ghosts

        # Encourage exploration when safe
        if not ghost_dists or min(ghost_dists) > 5:
            features['explore'] = 1

        # Penalize stopping to keep moving
        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        is_powered = my_state.scared_timer > 0

        if is_powered:  # Weights when powered (chasing mode)
            return {
                'successor_score': 100,      # Still prioritize food
                'distance_to_food': -1,      # Seek food
                'distance_to_power_dot': -100,  # Continue seeking power dots if available
                'distance_to_invader': -50,  # Strongly chase invaders
                'stop': -100                # Avoid stopping
            }
        else:  # Weights when not powered (escaping/food-seeking mode)
            return {
                'successor_score': 100,      # Prioritize food when safe
                'distance_to_food': -1,      # Seek regular food when safe
                'distance_to_power_dot': -100,  # Strongly prioritize power dots when available
                'escape_priority': 1000,     # Strongly prioritize escaping when ghost is within 3 moves
                'distance_to_ghost': 10,     # Moderate avoidance when ghost is 4-5 moves away
                'explore': 5,               # Encourage exploration when safe
                'stop': -100                # Avoid stopping
            }
        
class DefensiveHeuristicAgent(ReflexCaptureAgent):
    """Reflex agent that keeps its side Pacman-free and avoids powered enemies."""
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Defensive stance
        features['on_defense'] = 1 if not my_state.is_pacman else 0
        
        # Invader handling
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position()]
        features['num_invaders'] = len(invaders)
        if invaders:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Scared state handling (when enemy ate power pellet and we're white)
        is_scared = my_state.scared_timer > 0  # True if we're scared (white)
        if is_scared and invaders:
            min_invader_dist = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders])
            if min_invader_dist <= 5:  # Avoid if invader is close
                features['flee_invader'] = min_invader_dist
            else:
                features['flee_invader'] = 0  # Ignore if far away

        # Movement penalties
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        is_scared = my_state.scared_timer > 0

        if is_scared:  # Weights when scared (fleeing mode)
            return {
                'num_invaders': -1000,      # Still prioritize reducing invaders
                'on_defense': 100,          # Stay defensive
                'invader_distance': -5,     # Less aggressive chasing
                'flee_invader': 20,         # Strong incentive to increase distance from invaders
                'stop': -100,              # Avoid stopping
                'reverse': -2              # Avoid reversing
            }
        else:  # Weights when not scared (normal defense mode)
            return {
                'num_invaders': -1000,      # Strongly discourage invaders
                'on_defense': 100,          # Reward staying on defense
                'invader_distance': -10,    # Chase invaders
                'flee_invader': 0,          # Ignore fleeing when not scared
                'stop': -100,              # Avoid stopping
                'reverse': -2              # Avoid reversing
            }