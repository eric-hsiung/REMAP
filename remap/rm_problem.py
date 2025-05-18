import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
import aprel
import functools
import operator
import re

class OptionedCookieEnvironment:
    """
    Implements the deterministic version of the cookie environment.
    The original environment was presented in Icarte et al 2019,
    and it was a stochastic environment.

    For the optioned environment, we present the following options.
    We can use either the options or propositions to form the input
    alphabet. For now, we use the options to form the alphabet.
    Later, we will use the propositions to form the alphabet.

    We use the environment layout presented in Figure 1a:
    
      Y
      |
    B-W-R

    Cookie is in B,
    Button is in Y.
    """

    def __init__(self):
        """
        The options are:
        - Move up to North room
        - Move to South room
        - Move to East room
        - Move to West room
        - Interact with cookie
        - Interact with button
        """
        self.options = ("N","S","E","W","C","B")
        self.valid_room_values = tuple("BWYR")
        self.current_state = self.sample_initial_state()

        self.valid_room_options = {
            "W": tuple("WNE"),
            "B": tuple("EC"),
            "Y": tuple("BS"),
            "R": tuple("W"),
        }

    def sample_initial_state(self):
        return {
            "room": "W",
            "contains_cookie": False,
            "contains_button": False,
            "ate_cookie": False,
            "activated_button": False,
        }

    def sample_transition(self, state, option):
        """
        Given the current state and an option, generate the next state
        Return:
            If option is valid or not, and the next state
        """
        room = state["room"]
        next_state = dict(state)
        if option not in self.valid_room_options[room]:
            ## Handle invalid options
            return False, next_state
        else:
            if (room == "B" and option == "C"):
                if (state["contains_cookie"] == True and state["ate_cookie"] == False):
                    next_state["contains_cookie"] = False
                    next_state["ate_cookie"] = True
                    return True, next_state
                else:
                    return False, next_state
            elif (room == "B" and option == "E"):
                next_state["room"] = "W"
                next_state["contains_cookie"] = False
                return True, next_state
            elif (room == "Y" and option == "B"):
                ## NOTE: We assume the button stays, but is activated.
                if (state["contains_button"] == True and state["activated_button"] == False):
                    next_state["contains_button"] = True
                    next_state["activated_button"] = True
                    return True, next_state
                else:
                    return False, next_state
            elif (room == "Y" and option == "S"):
                next_state["room"] = "W"
                next_state["contains_button"] = False
                return True, next_state
            elif (room == "R" and option == "W"):
                next_state["room"] = "W"
                return True, next_state
            elif room == "W":
                ##    Y
                ##    |
                ##  B-W-R
                if option == "N":
                    next_state["room"] = "Y"
                    return True, next_state
                elif option == "W":
                    next_state["room"] = "B"
                    return True, next_state
                elif option == "E":
                    next_state["room"] = "R"
                    return True, next_state
                else:
                    return False, next_state
            else:
                return False, next_state

    def reward_model(self, state, option, next_state):
        ## NOTE: The base environment only returns 1 if the cookie has been eaten,
        ## and 0 otherwise
        return int(state["contains_cookie"] == True and state["ate_cookie"] == False and option == "C" and next_state["contains_cookie"] == False and next_state["ate_cookie"] == True)


class RegexEnvironment:

    def __init__(self, regex, alphabet):
        self.actions = alphabet
        self.regex = re.compile(regex)

    def sample_initial_state(self):
        return ""

    def sample_transition(self, s, a):
        return s + a

    def reward_model(self, seq):
        """
        Evaluates a sequence
        -- Returns either 0 or 1
        """
        return int(self.regex.fullmatch(seq) is not None)

class VectorModulusEnvironment:
    """
    This class describes the modulus environment
    """

    def __init__(self, num_states, reward_model=None, deterministic=False):
        self.divisor = num_states ## Tuple of states per dimension
        self.actions = []
        for idx in range(len(num_states)):
            self.actions.append(tuple( 1 if i == idx else 0 for i in range(len(num_states))))
            self.actions.append(tuple(-1 if i == idx else 0 for i in range(len(num_states))))
        self.actions = tuple(self.actions)
        self.states = tuple(itertools.product(*tuple(range(x) for x in self.divisor)))
        self.rng = np.random.default_rng()
        self.reward_model = None
        if reward_model is None:
            self.reward_model = self._reward_model
        else:
            self.reward_model = reward_model

        self.deterministic = deterministic

    def features(self, traj):
        num_actions = len(traj) - 1
        feat = np.zeros(len(self.states))
        for idx in range(num_actions):
            a, next_s = traj[idx+1]
            _, s      = traj[idx]
            feat[s] += 1
        return feat

    def sample_initial_state(self, prior=None):
        """
        Given a prior over self.states, returns a sample from self.states
        """

        if prior is None:
            ## Uniform over states
            return random.choice(self.states)
        else:
            return random.choices(self.states, weights=prior)

    def sample_transition(self, s, a, sz=None):
        return tuple(val % self.divisor[idx] for idx, val in enumerate(map(sum, zip(s,a))))

    def product_iter(self, iterable):
        return functools.reduce(operator.mul, iterable, 1)

    def _reward_model(self, s, a, sp):
        """
        Reward model is:
            (a,b,c,d): ((a + A*b) + A*B*c) + A*B*C*d
            (A,B,C,D)
        """
        sv = 0
        for idx, coord in enumerate(sp):
            if idx > 0:
                sv += self.product_iter(self.divisor[:idx]) * coord
            else:
                sv += coord
        return sv

class ModulusEnvironment:
    """
    This class describes the modulus environment
    """

    def __init__(self, num_states, reward_model=None, deterministic=False):
        self.divisor = num_states
        self.actions = tuple((-2,-1,0,1,2))
        self.states = tuple(range(self.divisor))
        self.rng = np.random.default_rng()
        self.reward_model = None
        if reward_model is None:
            self.reward_model = self._reward_model
        else:
            self.reward_model = reward_model

        self.deterministic = deterministic

    def features(self, traj):
        num_actions = len(traj) - 1
        feat = np.zeros(len(self.states))
        for idx in range(num_actions):
            a, next_s = traj[idx+1]
            _, s      = traj[idx]
            feat[s] += 1
        return feat

    def sample_initial_state(self, prior=None):
        """
        Given a prior over self.states, returns a sample from self.states
        """

        if prior is None:
            ## Uniform over states
            return random.choice(self.states)
        else:
            return random.choices(self.states, weights=prior)

    def transition_distribution(self, s, a):
        """
        Given a state and an action, returns a probability distribution over the next possible states.
        """

        prob = np.zeros(self.divisor)
        if a == 0:
            ## 0.5 probability of staying in current state; the remainder is uniform over the others
            prob[:] = 0.5/(self.divisor-1)
            prob[s] = 0.5
        elif a == 1:
            ## 0.8 of advancing to (s+1)%d
            ## 0.2 of staying in s
            t = (s+1) % self.divisor
            prob[s] = 0.2
            prob[t] = 0.8
        elif a == 2:
            ## 0.6 of advancing to (s+2)%d
            ## 0.2 of advancing to (s+1)%d
            ## 0.2 of staying in s
            u = (s+2) % self.divisor
            t = (s+1) % self.divisor
            prob[s] = 0.2
            prob[t] = 0.2
            prob[u] = 0.6
        elif a == -1:
            ## 0.8 of advancing to (s-1)%d
            ## 0.2 of staying in s
            t = (s-1) % self.divisor
            prob[s] = 0.2
            prob[t] = 0.8
        elif a == -2:
            ## 0.6 of advancing to (s+2)%d
            ## 0.2 of advancing to (s+1)%d
            ## 0.2 of staying in s
            u = (s-2) % self.divisor
            t = (s-1) % self.divisor
            prob[s] = 0.2
            prob[t] = 0.2
            prob[u] = 0.6
        else:
            raise ValueError
        return prob

    def sample_transition(self, s, a, sz=None):
        if not self.deterministic:
            prob = self.transition_distribution(s,a)
            if sz is None:
                return self.rng.choice(self.divisor, size=sz, p=prob)
            else:
                raise NotImplementedError
        else:
            return (s + a) % self.divisor

    def _reward_model(self, s, a, sp):
        return sp

