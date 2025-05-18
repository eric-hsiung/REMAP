import queue

class Problem:

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state
        """
        return state in self.goal_state_set

class AutomataProblem(Problem):

    def __init__(self, rm):
        super().__init__(rm.u0, rm.terminals)
        self.reward_machine = rm

    def get_actions(self, state):
        return set(formula for next_state, formula in self.reward_machine.delta_u[state].items())

    def execute(self, state, action):
        for next_state, formula in self.reward_machine.delta_u[state].items():
            if action == formula:
                return next_state

    def expand(self, node):
        for action in self.get_actions(node.state):
            yield Node.child_node(self, node, action)

    def step_cost(self, state, action):
        return 1

    def iterative_deepening_search(self, limit=100):
        d = 0
        all_solutions = []
        while d <= limit:
            solutions_list = self.depth_limited_search(d)
            all_solutions.extend(solutions_list)
            d = d + 1

        return all_solutions

    def depth_limited_search(self, depth):
        """
        Returns: (solution, Failure Bool, Cutoff Bool)
            solution = a list of (s,a) pairs, with the first s as the initial state, and the last s as the goal state
            Failure Bool = True if failure, False if success
            Cutoff Bool = True if the cutoff was activated, False by default
        """
        node = Node(self.initial_state, None, None, 0)
        frontier = Frontier(queue.LifoQueue())
        frontier.insert(node)
        explored = set()

        result = []
        failure = False
        cutoff = False

        possible_solutions = []

        while not frontier.empty():
            node = frontier.pop()
            if self.is_goal(node.state) or node.depth() > depth:
                possible_solutions.append(node.solution())
            else:
                explored.add(node)
                for child in self.expand(node):
                    if child not in explored and child not in frontier:
                        frontier.insert(child)
        return possible_solutions

class Frontier:
    def __init__(self, queue_ds):
        self.queue = queue_ds
        self.set = set()

    def empty(self):
        return self.queue.empty()

    def pop(self):
        el = self.queue.get(block=False)
        self.set.discard(el)
        return el

    def insert(self, el):
        self.set.add(el)
        self.queue.put_nowait(el)

    def __contains__(self, el):
        return el in self.set

class Node:

    def __init__(self, state: tuple, parent, action: tuple, path_cost):
        self.state = state ## State that this Node corresponds to
        self.parent = parent ## The parent Node of this Node
        self.action = action ## The action that was taken from the parent Node to this node
        self.path_cost = path_cost ## Total path cost from the initial state to this node

    def depth(self):
        return self.path_cost

    def solution(self):
        """
        Returns the path from the root node to this node in a list
        containing (s, a) of state and the action taken from that state
        """
        p = []
        node = self
        stack_state = queue.LifoQueue()
        stack_action = queue.LifoQueue()
        while node is not None:
            stack_state.put(node.state)
            stack_action.put(node.action)
            node = node.parent

        if not stack_state.empty():
            s = stack_state.get() ## Get the initial state
            a = stack_action.get() ## The action on the top should be None

        while not stack_action.empty():
            a = stack_action.get()
            p.append((s,a))
            s = stack_state.get()

        p.append((s, None))
        return p

    @staticmethod
    def child_node(problem: Problem, parent, action: tuple):
        state = problem.execute(parent.state, action)
        return Node(state, parent, action, parent.path_cost + problem.step_cost(parent.state, action))

    def __hash__(self):
        return hash(self.state)

    ## For a priority queue, we need to define comparison functions
    def __lt__(self, other):
        return self.path_cost <  other.path_cost
    def __le__(self, other):
        return self.path_cost <= other.path_cost
    def __gt__(self, other):
        return self.path_cost >  other.path_cost
    def __ge__(self, other):
        return self.path_cost >= other.path_cost
    def __eq__(self, other):
        return self.path_cost == other.path_cost
    def __ne__(self, other):
        return self.path_cost != other.path_cost
