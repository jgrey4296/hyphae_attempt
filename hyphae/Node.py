from uuid import uuid1
from . import constants
from .constants import randf
from cairo_utils.math import get_distance, get_normal, clamp
import numpy as np
import numpy.random as rand
import IPython

class Node:
    """ A Single Point in the Hyphae Calculation """

    def __init__(self, loc, d, distance, perpendicular=False, colour=None):
        assert(len(loc) == 2)
        if colour is None:
            colour = constants.NODE_COLOUR()
            colour[3] = constants.NODE_ALPHA
        self.id = uuid1()
        self.loc = loc
        self.d = d #diameter
        #distance from branch:
        self.distance = distance
        self.remaining = constants.MAX_ATTEMPTS_PER_NODE
        self.perpendicular = perpendicular
        self.colour = colour
        self.bbox = np.row_stack((self.loc - self.d,
                                  self.loc + self.d))

        self.size_decay = constants.NODE_SIZE_DECAY

        self.backtrack_likelihood = constants.BACKTRACK_LIKELIHOOD
        
        #delta in space
        self.delta = constants.NEIGHBOUR_DELTA
        #Wiggle in rads
        self.wiggle_chance= constants.WIGGLE_CHANCE
        self.wiggle_amnt = constants.WIGGLE_AMNT
        self.wiggle_variance = constants.WIGGLE_VARIANCE

        #split in rads
        self.split_chance = constants.SPLIT_CHANCE
        self.split_angle = constants.SPLIT_ANGLE
        self.split_variance = constants.SPLIT_ANGLE_VARIANCE

    def mutate(self, loc, d, distance,
               mod_chances=(0,0,0,0)):
        #return a new node, with mutated values
        newNode = Node(loc, d, distance, colour=self.colour)
        
        #mutate colour?
        if randf() < mod_chances[0]:
            mod = np.interp(randf(4), (0,1), constants.mut_c_range)
            #don't mod alpha
            alpha = self.colour[3]
            #clamp
            new_colour = np.interp(self.colour + mod, [0.1,1],[0.1,1])
            new_colour[3] = alpha
            newNode.colour = new_colour
        else:
            #if not modifying, copy
            newNode.colour = self.colour
            
        #mutate delta?
        if randf() < mod_chances[1]:
            mod = np.interp(randf(), (0, 1), constants.mut_d_range)
            newDelta = clamp(self.delta + mod, *constants.DELTA_CLAMP)
            newNode.delta = newDelta
        else:
            newNode.delta = self.delta
            
        #mutate wiggle?
        if randf() < mod_chances[2]:
            #chance
            wiggle_chance_mod = np.interp(randf(), (0,1), constants.mut_wcr)
            newNode.wiggle_chance = clamp(self.wiggle_chance + wiggle_chance_mod)
            #amnt
            wiggle_amnt_mod = np.interp(randf(), (0, 1), constants.mut_war)
            newNode.wiggle_amnt = constants.rad_clamp(self.wiggle_amnt + wiggle_amnt_mod)
            #var
            wiggle_var_mod = np.interp(randf(), (0,1), constants.mut_wvr)
            newNode.wiggle_variance = constants.rad_clamp(self.wiggle_variance + wiggle_var_mod) 
        else:
            newNode.wiggle_chance = self.wiggle_chance
            newNode.wiggle_amnt = self.wiggle_amnt
            newNode.wiggle_variance = self.wiggle_variance
            

        #mutate split?
        if randf() < mod_chances[3]:
            #chance
            #chance
            split_chance_mod = np.interp(randf(), (0,1), constants.mut_scr)
            newNode.split_chance = clamp(self.split_chance + split_chance_mod)
            #amnt
            split_angle_mod = np.interp(randf(), (0, 1), constants.mut_sar)
            newNode.split_angle = constants.rad_clamp(self.split_angle + split_angle_mod)
            #var
            split_var_mod = np.interp(randf(), (0,1), constants.mut_svr)
            newNode.split_variance = constants.rad_clamp(self.split_variance + split_var_mod) 
        else:
            newNode.split_chance = self.split_chance
            newNode.split_angle = self.split_angle
            newNode.split_variance = self.split_variance
            
        if randf() < mod_chances[4]:
            newBackTrackChance = np.interp(randf(), (0,1), constants.mut_btc)
            newNode.backtrack_likelihood = clamp(self.backtrack_likelihood + newBackTrackChance)
        else:
            newNode.backtrack_likelihood = self.backtrack_likelihood
            
        return newNode
        
    def get_delta_bbox(self):
        return self.d + self.delta
                    
    def distance_to(self, point):
        return get_distance(self.loc, point)

    def distance_to_node(self, node):
        assert(isinstance(node, Node))
        return get_distance(self.loc, node.loc)
    
    def get_normal(self, other):
        assert(isinstance(other, Node))
        return get_normal(self.loc, other.loc)

    def get_random_vec(self):
        return get_normal(self.loc, self.loc + (np.random.random(2) - 0.5))

    def move(self, dir=None):
        if dir is None:
            dir = self.get_random_vec()
        return self.loc + (dir * (2 * self.d))

    def attempt(self):
        if self.open():
            self.remaining -= 1

    def open(self):
        return bool(self.remaining)

    def force_open(self):
        self.remaining = constants.MAX_ATTEMPTS_PER_NODE
    
    def able_to_branch(self):
        return self.distance > constants.MIN_BRANCH_LENGTH
