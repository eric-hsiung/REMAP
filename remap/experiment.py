from collections import deque
import itertools
import argparse
import random
import pytest
import numpy as np
import networkx as nx
from lstar import BijectiveIndexMapping, TableList, EquivalenceClass, SymbolicObservationTable
from lstar import get_vars, concat, symbolic_lstar, range_prefixes
from rm_problem import ModulusEnvironment, VectorModulusEnvironment, RegexEnvironment, OptionedCookieEnvironment
from reward_machines.reward_machine import RewardMachine
from reward_machines.reward_machine_utils import evaluate_dnf
from search import AutomataProblem
from Mealy import Mealy
from Moore import Moore
import z3
import exrex
import sympy
import os

class Agent:
    """
    Implements various RL algorithms for computing policies
    """

    def __init__(self, env):
        """
        NOTE: The env Environment needs to have the following attributes:
         .states
         .actions
         .transition_distribution(s,a)[s'] (Returns as dict of next state probabilities)
            ^^^ This can easily be obtained from a hypothesis
         .reward_model(s,a,s')
            ^^^ This can easily be read from the observation table (row -> action -> row)
                with reward set
        """
        pass

    def evaluate_policy(self, pi, V, threshold):
        """
        Evaluates a policy, given a deterministic policy
        The value of the policy is considered converged once the largest change in subsequent
        evluations has reach below a certain threshold
        """
        V_old = np.copy(V)
        V_new = np.zeros(len(self.env.states))
        while True:
            max_error = 0
            for s in self.env.states:
                a = pi[s]
                v = V_old[s]
                V_new[s] = 0
                for sp in self.env.states:
                    V_new[s] += self.env.transition_distribution(s, a)[sp]*(self.env.reward_model(s, a, sp) + self.gamma * V_old[sp])
                max_error = max(max_error, np.abs(v - V_new[s]))
            #print(f"  --> max_error: {max_error}, V_new: {V_new}")
            V_old[:] = V_new
            if max_error < threshold:
                break
        return V_new

    def policy_improvement(self, pi, V):
        """
        Returns true or false. If false is returned, then we need to keep iterating
        Here, the policy pi is updated in place
        """
        converged = True
        for s in self.env.states:
            a_old = pi[s]
            max_val = None
            argmax = None
            for a in self.actions: 
                val = 0
                for sp in self.env.states:
                    val += self.env.transition_distribution(s, a)[sp]*(self.env.reward_model(s, a, sp) + self.gamma * V[sp])
                if max_val is None:
                    max_val = val
                    argmax = a
                else:
                    if val > max_val:
                        max_val = val
                        argmax = a
            pi[s] = argmax
            if a_old != pi[s]:
                converged = False
        return converged
        
    def policy_iteration(self, threshold=1e-3):
        """
        Performs vanilla policy iteration using policy evaluation and policy improvement
        Policy improvement adjusts the policy in-place
        """
        states = self.env.states
        actions = self.actions
        ## We want to learn a policy pi: S -> A for all states in S
        pi = self.env.rng.choice(actions, len(states)) ## Initial random policy
        V = np.zeros(len(self.env.states))
        converged = False
        while not converged:
            V = self.evaluate_policy(pi, V, threshold)
            converged = self.policy_improvement(pi, V)
        return pi

class RewardMachineTeacher:
    """
    NOTE: In Icarte 2018, these are specified as `simple reward machines';
    that is, they are Mealy machines, with real constants as the reward functions.
    
    For now, we implement the OfficeWorld reward machines as in Icarte 2018.

    We observe that for the OfficeWorld domain, and also for the minecraft domain,
    only a single proposition can be true at a time; individual propositions are represented
    by individual letters. Those domains process true_props, a str, that contains only
    letters of propositions which are true (if letters don't appear in true_props, then it
    is assumed that those propositions are false).

    """
    class EnvWrapper:

        def __init__(self, moore_rm):
            self.moore_rm = moore_rm

        def sample_initial_state(self):
            return self.moore_rm.initial_state
        
        def sample_transition(self, s, a):
            return self.moore_rm.transitions[s][a]

        def reward_model(self, seq):
            """
            Since these are defined as Mealy machines, what reward would we get at the end of the sequence?
            If the sequence is k units long, we use the first k-1 units to get to some state q in the machine.
            Then the kth element of the sequence will be associated with a reward in the machine:
            (q, kth) -> reward

            NOTE: We've converted these to use Moore machines.
            """
            q = self.sample_initial_state()
            max_idx = len(seq)
            done = False
            reward = self.moore_rm.output_table[q]
            for idx in range(max_idx):
                a = seq[idx]
                ## If q is not the terminal state
                next_q = self.sample_transition(q, a)
                reward = self.moore_rm.output_table[next_q]
                if next_q == "HALT":
                    ## Once we reach the absorbing state, we're guaranteed to be stuck there
                    break
                else:
                    q = next_q
            if reward is None:
                reward = 0
            return int(reward)

    def __init__(self, rm_file, propositions, seq_sample_size, init_state=None, forced_test=None, sophisticated_sampling=False):
        """
        propositions is a str; the individual letters in the string are propositions
        """
        self.incomplete_mealy_rm = RewardMachine(rm_file)
        transitions, alphabet, output_alphabet = self.rm_to_complete_mealy(self.incomplete_mealy_rm, propositions)
        states = list(self.incomplete_mealy_rm.U)
        states.append(self.incomplete_mealy_rm.terminal_u)
        self.mealy_rm = Mealy(states, alphabet, output_alphabet, transitions, self.incomplete_mealy_rm.u0)
        self.moore_rm, self.needs_additional_output = self.mealy_rm.to_moore()
        print("First Moore")
        print(self.moore_rm)
        self.halting_value = -100
        self.moore_rm.add_halting_state(self.halting_value)
        print("Halted Moore")
        print(self.moore_rm)
       
        ## Choose either random sample sequences or sophisticated sampling
        self.use_sophisticated_sampling = sophisticated_sampling

        if self.use_sophisticated_sampling:
            self.sample_sequences = self.ids_sample_sequences
        else:
            self.sample_sequences = self.geometric_sample_sequences
 
        self.env = RewardMachineTeacher.EnvWrapper(self.moore_rm)
        self.q0 = self.env.sample_initial_state()
        ## The input alphabet should be the power set of the input alphabet
        self.sigma_I = self.powerset(propositions)
        ## The output alphabet should be the possible reward values that can be returned from the reward machine
        self.sigma_O = tuple(self.moore_rm.output_alphabet) ## NOTE: the initial and HALTING states need to be treated separately
        self.rng = np.random.default_rng()
        self.seq_sample_size = seq_sample_size
        self.forced_seq = None
        if forced_test is not None:
            self.forced_seq = forced_test

    def rm_to_complete_mealy(self, rm, base_alphabet):
        """
        We convert the input reward machine to the complete Mealy machine.
        Note that edges in the original RM are symbolic. So we can either keep all
        edges symbolic, or we can expand all the symbolic edges explicitly.

        In this version, we expand all the symbolic edges to each edge where the sets
        of propositions are listed.

        Note that in the symbolic version, if we have a set of propositions,
        then if \phi_1 and \phi_2 are expressions over the propositions, then
        !(\phi_1 or \phi_2) captures the remaining cases that are not explicitly specified.
        """
        transitions = {}
        alphabet = self.powerset(base_alphabet)
        output_alphabet = set()
        ## NOTE: According to Icarte 2018, "if no valid transition is available using the true propositions
        ##       then we automatically transition to the terminal state"
        ## Additionally, delta_u does NOT contain the terminal state. If we transition to the terminal state
        ## due to no other option, then we obtain a reward of zero, according to Icarte 2018.
        for cur_state, edges in rm.delta_u.items():
            for next_state, formula in edges.items():
                output = int(rm.delta_r[cur_state][next_state].get_reward(None))
                output_alphabet.add(output)
                ## Check the powerset
                for alpha in alphabet:
                    if evaluate_dnf(formula, alpha):
                        if cur_state not in transitions:
                            transitions[cur_state] = {alpha: (next_state, output)}
                        else:
                            transitions[cur_state][alpha] = (next_state, output)
        ## Check for completeness in all the states with transitions. If any states have incomplete transitions,
        ## send the remaining transitions to the terminal state, along with a reward of 0
        for cur_state in transitions:
            for alpha in alphabet:
                if alpha not in transitions[cur_state]:
                    transitions[cur_state][alpha] = (rm.terminal_u, 0)
            
        return transitions, alphabet, tuple(output_alphabet)

    def powerset(self, s):
        ## Powerset recipe is from https://docs.python.org/3/library/itertools.html#itertools-recipes
        return tuple("".join(x) for x in itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))

    def ids_sample_sequences(self, quantity):
        """
        Sample via Iterative Deepening Search
        """
        ## Geometric distribution sequences
        alphabet = self.sigma_I
        geometric_sequences = []
        
        ## Ground truth RM sequences
        ground_truth_rm = self.incomplete_mealy_rm
        num_states = ground_truth_rm.count_states()
        ## First, generate all possible paths of length at most num_states
        ap = AutomataProblem(ground_truth_rm)
        all_solutions = ap.iterative_deepening_search(num_states)
        cached_options = dict() ## Formula to list of propositions satisfying the formula

        positive_sampled_sequences = list()

        for solution in all_solutions:
            for _ in range((len(solution)-1)*len(alphabet)*quantity):
                new_sequence = list()
                for state, formula in solution:
                    if formula is not None:
                        ## Sample propositions which would make the formula true
                        if formula not in cached_options:
                            ## sample from cached_options
                            satisfying_propositions = list(alpha for alpha in alphabet if evaluate_dnf(formula, alpha))
                            cached_options[formula] = satisfying_propositions

                        ## Sample propositions making the formula true
                        new_sequence.append(random.choice(cached_options[formula]))
                sequence = tuple(new_sequence)
                positive_sampled_sequences.append(sequence)
        geometric_sequences.extend(positive_sampled_sequences)

        geometric_sequences.extend(self.geometric_sample_sequences(quantity))

        return geometric_sequences

    def geometric_sample_sequences(self, quantity):
        """
        We can do random sequences, or we can sample sequences from the reward machine itself
        Lets do random sequences first
        """
        p = 0.2
        if self.forced_seq is not None:
            quantity = quantity - len(self.forced_seq) - 1
        else:
            quantity = quantity - 1

        seq_lengths = self.rng.geometric(p, quantity)
        sequences = []
        sequences.append(tuple())
        if self.forced_seq is not None:
            for seq in self.forced_seq:
                sequences.append(seq)
        for length in seq_lengths:
            ## Generate a random sequence of the desired length
            sequences.append(tuple(random.choices(self.sigma_I, k=length)))
        return sequences

    def evaluate_sequence(self, sequence):
        return self.env.reward_model(sequence)

    def preference_query(self, s1, s2):
        #print(f"<< PREF Q: {s1}")
        #print(f"<< PREF Q: {s2}")
        r1 = self.evaluate_sequence(s1)
        r2 = self.evaluate_sequence(s2)
        if r1 == r2:
            return 0
        elif r1 > r2:
            return 1
        else:
            return -1

    def __equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Computes an empirical equivalence query

        Observed Cases of equiv query:
            Init State = 3; Test Seq: (0, -2, 2)
        """
        def evaluate_hypothesis(seq):
            #print(delta)
            q = init_state
            for a in seq:
                q = delta[q][a]
            return output_fnc[q]

        sequences = self.sample_sequences(self.seq_sample_size)

        #print("=== >>> Teacher Equiv Query <<< ===\n")
        #print(f"  Teacher Init State: {self.q0}\n")
        for seq in sequences:
            teacher_output = self.evaluate_sequence(seq)
            learner_output = evaluate_hypothesis(seq)
            #print(f"  SEQ LEN: {len(seq)}  --  SEQ: {seq}\n   > TEACHER OUT: {teacher_output}\n   > LEARNER OUT: {learner_output}")

            if teacher_output != learner_output:
                return False, (seq, teacher_output)

        return True, None

    def equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Performs a symbolic evaluation of whether two Moore machines are equivalent.

        This implements an optimized version of the Hopcroft-Karp algorithm. https://arxiv.org/abs/0907.5058

        Also here, have adopted the equivalence check from https://github.com/caleb531/automata/blob/main/automata/fa/dfa.py

        The input values in the function refer to the automata to be tested. The ground truth is the teacher's automata.
        """
        print(states)
        print(sigma_I)
        print(sigma_O)
        print(init_state)
        print(delta)
        print(output_fnc)

        if sigma_I != self.sigma_I:
            return ValueError("The teacher and learner have different input alphabets")

        initial_state_teacher = self.q0
        initial_state_learner = init_state

        ## In all operations below, we assume that the teacher is on the left side and the learner is on the right side.
        state_sets = nx.utils.union_find.UnionFind((initial_state_teacher, initial_state_learner))
        pair_stack = deque()

        state_sets.union( (1, initial_state_teacher, initial_state_learner), (0, initial_state_learner, initial_state_teacher))
        pair_stack.append((initial_state_teacher, initial_state_learner, tuple()))

        ## NOTE: This algorithm visits the states in both automata in a depth-first order. If the ordering of all the
        ## output values of the algorithm match, then both automata are equivalent. However, we will need to obtain a
        ## counterexample in the case that the automata are found inequivalent. This means that we will need to perform
        ## a DFS to the inequivalent state and record the sequence that gets us there. And return that as the cex.
        while pair_stack:
            q_T, q_L, cex = pair_stack.pop()
            print(f"Comparing {q_T} and {q_L}")

            ## NOTE: that HALT is an absorbing state in the teacher's Moore machine
            ## If the values are not the same, then the automata cannot be equivalent
            teacher_output = int(self.env.moore_rm.output_table[q_T])
            if output_fnc[q_L] != teacher_output:
                return False, (cex, teacher_output)

            for letter in sigma_I:
                ## Here, r_T and r_L refer to set representatives
                n_T = self.env.moore_rm.transitions[q_T][letter]
                n_L = delta[q_L][letter]
                r_T = state_sets[(1, n_T, n_L)]
                r_L = state_sets[(0, n_L, n_T)]

                if r_T != r_L:
                    seq = list(cex)
                    seq.append(letter)
                    state_sets.union(r_T, r_L)
                    pair_stack.append((n_T, n_L, tuple(seq)))
        return True, None

class OptionedCookieEnvironmentTeacher:

    def __init__(self, seq_sample_size, init_state=None, forced_test=None):
        self.env = OptionedCookieEnvironment()
        self.q0 = self.env.sample_initial_state() ## Select an initial state
        ## We need to confirm whether the algorithm will learn the naive RM or the perfect RM
        ## NOTE: We can learn the perfect RM if we explicitly specify the number of states we expect there
        ## to be, so this depends on how we defined the output alphabet.
        ## If the output alphabet is simply just reward, then there is a possibility that the learner learns
        ## the naive RM.
        ## If we give the explicity (state id, reward value), then the learner should learn the perfect RM.
        self.sigma_I = self.env.options
        self.sigma_O = tuple((-1, 0, 1)) ## Here we initialize with the reward only alphabet
        self.rng = np.random.default_rng()
        self.seq_sample_size = seq_sample_size
        self.forced_seq = None
        if forced_test is not None:
            self.forced_seq = forced_test

    def sample_sequences(self, quantity):
        """
        Uses a geometric distribution to sample a sequence.
        """
        p = 0.2
        if self.forced_seq is not None:
            quantity = quantity - len(self.forced_seq) - 1
        else:
            quantity = quantity - 1

        seq_lengths = self.rng.geometric(p, quantity)
        sequences = []
        sequences.append(tuple())
        if self.forced_seq is not None:
            for seq in self.forced_seq:
                sequences.append(seq)
        for length in seq_lengths:
            ## Generate a random sequence of the desired length
            sequences.append(tuple(random.choices(self.sigma_I, k=length)))
        return sequences

    def evaluate_sequence(self, sequence):
        """
        Convert sequences to strings
        """
        state = self.q0
        total_return = 0
        for option in sequence:
            is_valid, next_state = self.env.sample_transition(state, option)
            if is_valid:
                reward = self.env.reward_model(state, option, next_state)
                total_return += reward
                state = next_state
            else:
                return -1
        return total_return

    def preference_query(self, s1, s2):
        print(f"<< PREF Q: {s1}")
        print(f"<< PREF Q: {s2}")
        r1 = self.evaluate_sequence(s1)
        r2 = self.evaluate_sequence(s2)
        if r1 == r2:
            return 0
        elif r1 > r2:
            return 1
        else:
            return -1

    def equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Computes an empirical equivalence query

        Observed Cases of equiv query:
            Init State = 3; Test Seq: (0, -2, 2)
        """
        def evaluate_hypothesis(seq):
            print(delta)
            q = init_state
            for a in seq:
                q = delta[q][a]
            return output_fnc[q]

        sequences = self.sample_sequences(self.seq_sample_size)

        print("=== >>> Teacher Equiv Query <<< ===\n")
        print(f"  Teacher Init State: {self.q0}\n")
        for seq in sequences:
            teacher_output = self.evaluate_sequence(seq)
            learner_output = evaluate_hypothesis(seq)
            print(f"  SEQ LEN: {len(seq)}  --  SEQ: {seq}\n   > TEACHER OUT: {teacher_output}\n   > LEARNER OUT: {learner_output}")

            if teacher_output != learner_output:
                return False, (seq, teacher_output)

        return True, None
    


class MultiRegexEnvironmentTeacher:

    class EnvWrapper:

        def __init__(self, regex_list, alphabet):
            self.env_list = list(RegexEnvironment(regex, alphabet) for regex in regex_list)
            self.actions = alphabet

        def sample_initial_state(self):
            return ""
        
        def sample_transition(self, s, a):
            return s + a

        def reward_model(self, seq):
            """
            Tests the sequences in order of the regexes. First match wins reward = 1*index of the regex
            """

            for idx, env in enumerate(self.env_list):
                if 1 == env.reward_model(seq):
                    return idx + 1
            return 0

    def __init__(self, regex_list, alphabet, seq_sample_size, init_state=None, forced_test=None):
        """
        Allows teacher to teach multiple regexes to the student
        """
        self.env = MultiRegexEnvironmentTeacher.EnvWrapper(regex_list, alphabet)
        if init_state is None:
            self.q0 = self.env.sample_initial_state() ## Select an initial state
        else:
            self.q0 = init_state
        self.sigma_O = tuple(range(len(regex_list)+1))
        self.sigma_I = self.env.actions
        self.rng = np.random.default_rng()
        self.seq_sample_size = seq_sample_size
        self.forced_seq = None
        if forced_test is not None:
            self.forced_seq = forced_test

    def sample_sequences(self, quantity):
        """
        Uses a geometric distribution to sample a sequence.
        """
        p = 0.2
        if self.forced_seq is not None:
            quantity = quantity - len(self.forced_seq) - 1
        else:
            quantity = quantity - 1

        seq_lengths = self.rng.geometric(p, quantity)
        sequences = []
        sequences.append(tuple())
        if self.forced_seq is not None:
            for seq in self.forced_seq:
                sequences.append(seq)
        for length in seq_lengths:
            ## Generate a random sequence of the desired length
            sequences.append(tuple(random.choices(self.sigma_I, k=length)))
        return sequences

    def sequence_value(self, s):
        """
        Computes sum of rewards for this sequence, under ground truth reward
        """
        return sum(self.evaluate_sequence(seq) for seq in range_prefixes(s))

    def evaluate_sequence(self, s):
        """
        Convert sequences to strings
        """
        return self.env.reward_model("".join(s))

    def accuracy(self, s, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Returns 1 if s is classified correctly
        Returns 0 if s is classified incorrectly
        """
        def evaluate_hypothesis(seq):
            q = init_state
            for a in seq:
                q = delta[q][a]
            return output_fnc[q]

        teacher_output = self.evaluate_sequence(s)
        learner_output = evaluate_hypothesis(s)

        return int(teacher_output == learner_output)

    def preference_query(self, s1, s2):
        print(f"<< PREF Q: {s1}")
        print(f"<< PREF Q: {s2}")
        r1 = self.evaluate_sequence(s1)
        r2 = self.evaluate_sequence(s2)
        if r1 == r2:
            return 0
        elif r1 > r2:
            return 1
        else:
            return -1

    def equivalence_query_policy_eval(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Computes an empirical equivalence query via policy evaluation using the following strategy:
            - First, the teacher computes the optimal policies \pi^H and \pi^* under the
              hypothesis reward machine and ground truth reward, respectively.
            - Then, the teacher evaluates both \pi^H and \pi^* under the ground truth reward, R to
              obtain the average regret. Regret is the difference between the V^{\pi^*} values and
              the V^{\pi^H} values; in other words it is the different in expected returns between
              the optimal policy and some other policy.
                - If the average regret is zero, then the teacher concludes that the learner is
                  correct.
                - If the average regret is non-zero, then find a sequence s_H~\pi^H and s_*~\pi^*
                  where (wrt to the ground truth reward R) the return of s_H != the return of s_*.
                    - We write val_R(s) to denote the return of a sequence with respect to the
                      ground truth reward R, and val_H(s) to denote the return of a sequence with
                      respect to the hypothesis reward machine H.
                  The teacher then returns the counterexample CEX in the following form:
                    - CEX = (s_H, s_*), implying that val_R(s_H) > val_R(s_*), or
                    - CEX = (s_*, s_H), implying that val_R(s_*) > val_R(s_H)
                  The learner interprets CEX as the following high-level constraint:
                    val_H(CEX[0]) > val_H(CEX[1]) at all times. That is, when sending constraints to
                    the SMT solver, the learner converts val_H(CEX[0]) > val_H(CEX[1]) to a constraint
                    in terms of the currently used variables in the table for the SMT solver to use
        """
        pass

    def equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Computes an empirical equivalence query

        Observed Cases of equiv query:
            Init State = 3; Test Seq: (0, -2, 2)
        """
        def evaluate_hypothesis(seq):
            q = init_state
            for a in seq:
                q = delta[q][a]
            return output_fnc[q]

        sequences = self.sample_sequences(self.seq_sample_size)

        print("=== >>> Teacher Equiv Query <<< ===\n")
        print(f"  Teacher Init State: {self.q0}\n")
        print(delta)
        for seq in sequences:
            teacher_output = self.evaluate_sequence(seq)
            learner_output = evaluate_hypothesis(seq)
            print(f"  SEQ LEN: {len(seq)}  --  SEQ: {seq}\n   > TEACHER OUT: {teacher_output}\n   > LEARNER OUT: {learner_output}")

            if teacher_output != learner_output:
                return False, (seq, teacher_output)

        return True, None

class SingleRegexEnvironmentTeacher:
    def __init__(self, regex, alphabet, seq_sample_size, init_state=None, forced_test=None):
        """
        Allows teacher to reach multiple regexes to the student
        """
        self.env = RegexEnvironment(regex, alphabet)
        if init_state is None:
            self.q0 = self.env.sample_initial_state() ## Select an initial state
        else:
            self.q0 = init_state
        self.sigma_O = (0,1)
        self.sigma_I = self.env.actions
        self.rng = np.random.default_rng()
        self.seq_sample_size = seq_sample_size
        self.forced_seq = None
        if forced_test is not None:
            self.forced_seq = forced_test

    def sample_sequences(self, quantity):
        """
        Uses a geometric distribution to sample a sequence.
        """
        p = 0.2
        if self.forced_seq is not None:
            quantity = quantity - len(self.forced_seq) - 1
        else:
            quantity = quantity - 1

        seq_lengths = self.rng.geometric(p, quantity)
        sequences = []
        sequences.append(tuple())
        if self.forced_seq is not None:
            for seq in self.forced_seq:
                sequences.append(seq)
        for length in seq_lengths:
            ## Generate a random sequence of the desired length
            sequences.append(tuple(random.choices(self.sigma_I, k=length)))
        return sequences

    def evaluate_sequence(self, s):
        """
        Convert sequences to strings
        """
        return self.env.reward_model("".join(s))

    def preference_query(self, s1, s2):
        print(f"<< PREF Q: {s1}")
        print(f"<< PREF Q: {s2}")
        r1 = self.evaluate_sequence(s1)
        r2 = self.evaluate_sequence(s2)
        if r1 == r2:
            return 0
        elif r1 > r2:
            return 1
        else:
            return -1

    def equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Computes an empirical equivalence query

        Observed Cases of equiv query:
            Init State = 3; Test Seq: (0, -2, 2)
        """
        def evaluate_hypothesis(seq):
            print(delta)
            q = init_state
            for a in seq:
                q = delta[q][a]
            return output_fnc[q]

        sequences = self.sample_sequences(self.seq_sample_size)

        print("=== >>> Teacher Equiv Query <<< ===\n")
        print(f"  Teacher Init State: {self.q0}\n")
        for seq in sequences:
            teacher_output = self.evaluate_sequence(seq)
            learner_output = evaluate_hypothesis(seq)
            print(f"  SEQ LEN: {len(seq)}  --  SEQ: {seq}\n   > TEACHER OUT: {teacher_output}\n   > LEARNER OUT: {learner_output}")

            if teacher_output != learner_output:
                return False, (seq, teacher_output)

        return True, None

class VectorModulusTeacher:
    def __init__(self, modulus_size, seq_sample_size, init_state=None, forced_test=None):
        self.env = VectorModulusEnvironment(modulus_size, deterministic=True)
        if init_state is None:
            self.q0 = self.env.sample_initial_state() ## Select an initial state
        else:
            self.q0 = init_state
        if isinstance(modulus_size, tuple):
            p = 1
            for el in modulus_size:
                p = p * el
            self.sigma_O = tuple(v for v in range(p))
        else:
            self.sigma_O = tuple(v for v in range(modulus_size))
        self.sigma_I = self.env.actions
        self.rng = np.random.default_rng()
        self.seq_sample_size = seq_sample_size
        self.forced_seq = None
        if forced_test is not None:
            self.forced_seq = forced_test

    def sample_sequences(self, quantity):
        """
        Uses a geometric distribution to sample a sequence.
        """
        p = 0.2
        if self.forced_seq is not None:
            quantity = quantity - len(self.forced_seq) - 1
        else:
            quantity = quantity - 1

        seq_lengths = self.rng.geometric(p, quantity)
        sequences = []
        sequences.append(tuple())
        if self.forced_seq is not None:
            for seq in self.forced_seq:
                sequences.append(seq)
        for length in seq_lengths:
            ## Generate a random sequence of the desired length
            sequences.append(tuple(random.choices(self.sigma_I, k=length)))
        return sequences

    def evaluate_sequence(self, s):
        q = self.q0
        for a in s:
            q = self.env.sample_transition(q, a)

        return self.env.reward_model(None, None, q)

    def preference_query(self, s1, s2):
        r1 = self.evaluate_sequence(s1)
        r2 = self.evaluate_sequence(s2)
        if r1 == r2:
            return 0
        elif r1 > r2:
            return 1
        else:
            return -1

    def equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Computes an empirical equivalence query

        Observed Cases of equiv query:
            Init State = 3; Test Seq: (0, -2, 2)
        """
        def evaluate_hypothesis(seq):
            print(delta)
            q = init_state
            for a in seq:
                q = delta[q][(a,)]
            return output_fnc[q]

        sequences = self.sample_sequences(self.seq_sample_size)

        print("=== >>> Teacher Equiv Query <<< ===\n")
        print(f"  Teacher Init State: {self.q0}\n")
        for seq in sequences:
            teacher_output = self.evaluate_sequence(seq)
            learner_output = evaluate_hypothesis(seq)
            print(f"  SEQ LEN: {len(seq)}  --  SEQ: {seq}\n   > TEACHER OUT: {teacher_output}\n   > LEARNER OUT: {learner_output}")

            if teacher_output != learner_output:
                return False, (seq, teacher_output)

        return True, None
class ModulusTeacher:
    def __init__(self, modulus_size, seq_sample_size, init_state=None, forced_test=None):
        self.env = ModulusEnvironment(modulus_size, deterministic=True)
        if init_state is None:
            self.q0 = self.env.sample_initial_state() ## Select an initial state
        else:
            self.q0 = init_state
        if isinstance(modulus_size, tuple):
            p = 1
            for el in modulus_size:
                p = p * el
            self.sigma_O = tuple(v for v in range(p))
        else:
            self.sigma_O = tuple(v for v in range(modulus_size))
        self.sigma_I = self.env.actions
        self.rng = np.random.default_rng()
        self.seq_sample_size = seq_sample_size
        self.forced_seq = None
        if forced_test is not None:
            self.forced_seq = forced_test

    def sample_sequences(self, quantity):
        """
        Uses a geometric distribution to sample a sequence.
        """
        p = 0.2
        if self.forced_seq is not None:
            quantity = quantity - len(self.forced_seq) - 1
        else:
            quantity = quantity - 1

        seq_lengths = self.rng.geometric(p, quantity)
        sequences = []
        sequences.append(tuple())
        if self.forced_seq is not None:
            for seq in self.forced_seq:
                sequences.append(seq)
        for length in seq_lengths:
            ## Generate a random sequence of the desired length
            sequences.append(tuple(random.choices(self.sigma_I, k=length)))
        return sequences

    def evaluate_sequence(self, s):
        q = self.q0
        for a in s:
            q = self.env.sample_transition(q, a)

        return self.env.reward_model(None, None, q)

    def preference_query(self, s1, s2):
        r1 = self.evaluate_sequence(s1)
        r2 = self.evaluate_sequence(s2)
        if r1 == r2:
            return 0
        elif r1 > r2:
            return 1
        else:
            return -1

    def equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Computes an empirical equivalence query

        Observed Cases of equiv query:
            Init State = 3; Test Seq: (0, -2, 2)
        """
        def evaluate_hypothesis(seq):
            print(delta)
            q = init_state
            for a in seq:
                q = delta[q][a]
            return output_fnc[q]

        sequences = self.sample_sequences(self.seq_sample_size)

        print("=== >>> Teacher Equiv Query <<< ===\n")
        print(f"  Teacher Init State: {self.q0}\n")
        for seq in sequences:
            teacher_output = self.evaluate_sequence(seq)
            learner_output = evaluate_hypothesis(seq)
            print(f"  SEQ LEN: {len(seq)}  --  SEQ: {seq}\n   > TEACHER OUT: {teacher_output}\n   > LEARNER OUT: {learner_output}")

            if teacher_output != learner_output:
                return False, (seq, teacher_output)

        return True, None
class SumsModulusTeacher:
    def __init__(self, modulus_size, seq_sample_size, init_state=None, forced_test=None):
        self.env = ModulusEnvironment(modulus_size, deterministic=True)
        self.use_sums = True
        if init_state is None:
            self.q0 = self.env.sample_initial_state() ## Select an initial state
        else:
            self.q0 = init_state
        if isinstance(modulus_size, tuple):
            p = 1
            for el in modulus_size:
                p = p * el
            self.sigma_O = tuple(v for v in range(p))
        else:
            self.sigma_O = tuple(v for v in range(modulus_size))
        self.sigma_I = self.env.actions
        self.rng = np.random.default_rng()
        self.seq_sample_size = seq_sample_size
        self.forced_seq = None
        if forced_test is not None:
            self.forced_seq = forced_test

    def sample_sequences(self, quantity):
        """
        Uses a geometric distribution to sample a sequence.
        """
        p = 0.2
        if self.forced_seq is not None:
            quantity = quantity - len(self.forced_seq) - 1
        else:
            quantity = quantity - 1

        seq_lengths = self.rng.geometric(p, quantity)
        sequences = []
        sequences.append(tuple())
        if self.forced_seq is not None:
            for seq in self.forced_seq:
                sequences.append(seq)
        for length in seq_lengths:
            ## Generate a random sequence of the desired length
            sequences.append(tuple(random.choices(self.sigma_I, k=length)))
        return sequences

    def evaluate_sequence(self, s):
        q = self.q0
        total = self.env.reward_model(None, None, q)
        for a in s:
            q = self.env.sample_transition(q, a)
            r = self.env.reward_model(None, None, q)
            total += r

        return total

    def preference_query(self, s1, s2):
        r1 = self.evaluate_sequence(s1)
        r2 = self.evaluate_sequence(s2)
        if r1 == r2:
            return 0
        elif r1 > r2:
            return 1
        else:
            return -1

    def equivalence_query(self, states, sigma_I, sigma_O, init_state, delta, output_fnc):
        """
        Computes an empirical equivalence query

        Observed Cases of equiv query:
            Init State = 3; Test Seq: (0, -2, 2)
        """
        def evaluate_hypothesis(seq):
            print(delta)
            q = init_state
            total = output_fnc[q]
            for a in seq:
                q = delta[q][a]
                r = output_fnc[q]
                total += r
            return total

        sequences = self.sample_sequences(self.seq_sample_size)

        print("=== >>> Teacher Equiv Query <<< ===\n")
        print(f"  Teacher Init State: {self.q0}\n")
        for seq in sequences:
            teacher_output = self.evaluate_sequence(seq)
            learner_output = evaluate_hypothesis(seq)
            print(f"  SEQ LEN: {len(seq)}  --  SEQ: {seq}\n   > TEACHER OUT: {teacher_output}\n   > LEARNER OUT: {learner_output}")

            if teacher_output != learner_output:
                return False, (seq, teacher_output)

        return True, None

def main():
    #teacher = ModulusTeacher(100, 200, init_state=3, forced_test=[(0,-2,2), (0, -2, 1, 0, 1, 1, -1, 2, -1)])
    #teacher = ModulusTeacher((3,5), 200)
    ## Tested "a*b", "b*a", "a*b*", "(ab)*"
    #teacher = SingleRegexEnvironmentTeacher("a*b*", ("a", "b", "c"), 200, forced_test=[tuple(el) for el in ["bbb","ab","aab","bbb"]])
    teacher = MultiRegexEnvironmentTeacher(["a*b", "b*a", "(ab)*", "(ba)*", "aabbaabb"], ("a", "b"), 500, forced_test=[tuple(el) for el in ["bbb","ab","aab","bbb"]])
    print(tuple(teacher.sigma_I))
    symbolic_lstar(teacher.sigma_I, teacher.sigma_O, teacher)
    #symbolic_lstar(tuple((el,) for el in teacher.sigma_I), teacher.sigma_O, teacher)

def multi_regex_experiment(regexes, alphabet, sequence_samples, initial_state, exp_file):
    teacher = MultiRegexEnvironmentTeacher(regexes, alphabet, sequence_samples)
    hypothesis, experimental_data = symbolic_lstar(teacher.sigma_I, teacher.sigma_O, teacher)
    num_pref_q, num_ineq, num_unique_seq, num_ECs, num_unique_table_vars, up_shape, lo_shape, num_equiv_q, cex_lengths, events = experimental_data


    ## Compute accuracy here:
    ## For each type of regex, compute say 200 samples.
    ## Compute total accuracy on regexes and on non-acceptable strings....
    regex_accuracy = []
    total_accuracy = 0.0
    total_correct = 0
    total_tested = 0
    for regex in regexes:
        num_correct = 0
        num_tested = 0
        for _ in range(200):
            test_str = tuple(exrex.getone(regex, limit=1000))
            result = teacher.accuracy(test_str, *hypothesis)
            num_tested += 1
            num_correct += result
        regex_accuracy.append(num_correct/num_tested)
        total_correct += num_correct
        total_tested += num_tested
    total_accuracy = total_correct / total_tested

    csv = True

    if not csv: 
        exp_file.write("Experimental Results:\n")
        exp_file.write(f"Number of States: {len(hypothesis[0])}\n")

        exp_file.write(f"Total Number of Preference Queries: {num_pref_q}\n")
        exp_file.write(f"Total Number of Equivalence Queries: {num_equiv_q}\n")
        exp_file.write(f"Total Number of Inequalities: {num_ineq}\n")
        exp_file.write(f"Total Number of Unique Sequences Tested: {num_unique_seq}\n")
        exp_file.write(f"Total Number of Equivalence Classes Generated: {num_ECs}\n")
        exp_file.write(f"Total Number of Unique Table Variables upon Termination: {num_unique_table_vars}\n")
        exp_file.write(f"Upper Table Dimensions upon Termination: {up_shape}\n")
        exp_file.write(f"Lower Table Dimensions upon Termination: {lo_shape}\n")
        exp_file.write(f"CEX Lengths: {cex_lengths}\n\n\n")
    else:
        exp_file.write(f"{len(hypothesis[0])}#{num_pref_q}#{num_equiv_q}#{num_ineq}#{num_ECs}#{num_unique_table_vars}#{num_unique_seq}#{up_shape}#{lo_shape}#{cex_lengths}")
        for regex in regex_accuracy:
            exp_file.write(f"#{regex}")
        exp_file.write(f"#{total_accuracy}#{events}\n")
        
        

def do_multi_regex_experiment():
    train_regexes = ["a*b", "b*a", "(ab)*", "(ba)*", "aabbaabb"]

    with open(f"multi_regex_exp.log", "a") as exp_f:
        for _ in range(10):
            multi_regex_experiment(train_regexes, ("a","b"), 500, None, exp_f)

def do_simple_regex_experiment():
    #train_regexes = ["a*b", "b*a", "(ab)*", "(ba)*"]
    alphabet = ("a","b")
    train_regexes = ["a*b"]
    sample_seq = 500
    with open(f"events_simple_regex_exp_1_sample_{sample_seq}_expanded_alphabet_{len(alphabet)}.csv", "a") as exp_f:
        exp_f.write("'Number of States'#'Num Pref Q'#'Num Equiv Q'#'Num Ineq'#'Num ECs'#'Num Unique Table Vars'#'Num Unique Sequences'#'Upper Dim'#'Lower Dim'#'CEX Lengths'")
        for regex in train_regexes:
            exp_f.write(f"#'Accuracy ({regex})'")
        exp_f.write("#'Total Accuracy'#'Events'\n")
        for _ in range(10):
            multi_regex_experiment(train_regexes, alphabet, sample_seq, None, exp_f)

def sums_modulus_experiment(total_states, sequence_samples, initial_state, exp_file):
    teacher = SumsModulusTeacher(total_states, sequence_samples, init_state=initial_state)
    hypothesis, experimental_data = symbolic_lstar(tuple((el,) for el in teacher.sigma_I), teacher.sigma_O, teacher)
    num_pref_q, num_ineq, num_unique_seq, num_ECs, num_unique_table_vars, up_shape, lo_shape, num_equiv_q, cex_lengths, events = experimental_data

    csv = True

    if not csv: 
        exp_file.write("Experimental Results:\n")
        exp_file.write(f"Number of States: {len(hypothesis[0])}\n")

        exp_file.write(f"Total Number of Preference Queries: {num_pref_q}\n")
        exp_file.write(f"Total Number of Equivalence Queries: {num_equiv_q}\n")
        exp_file.write(f"Total Number of Inequalities: {num_ineq}\n")
        exp_file.write(f"Total Number of Unique Sequences Tested: {num_unique_seq}\n")
        exp_file.write(f"Total Number of Equivalence Classes Generated: {num_ECs}\n")
        exp_file.write(f"Total Number of Unique Table Variables upon Termination: {num_unique_table_vars}\n")
        exp_file.write(f"Upper Table Dimensions upon Termination: {up_shape}\n")
        exp_file.write(f"Lower Table Dimensions upon Termination: {lo_shape}\n")
        exp_file.write(f"CEX Lengths: {cex_lengths}\n\n\n")
    else:
        exp_file.write(f"{len(hypothesis[0])}#{num_pref_q}#{num_equiv_q}#{num_ineq}#{num_ECs}#{num_unique_table_vars}#{num_unique_seq}#{up_shape}#{lo_shape}#{cex_lengths}#{tuple(events)}\n")

def do_sums_modulus_experiment():
    with open(f"sums_mod_exp.csv", "a") as exp_f:
        for sz in [5,7]:
            for _ in range(10):
                sums_modulus_experiment(sz, 200, 3, exp_f)


def modulus_experiment(total_states, sequence_samples, initial_state, exp_file):
    teacher = ModulusTeacher(total_states, sequence_samples, init_state=initial_state)
    hypothesis, experimental_data = symbolic_lstar(tuple((el,) for el in teacher.sigma_I), teacher.sigma_O, teacher)
    num_pref_q, num_ineq, num_unique_seq, num_ECs, num_unique_table_vars, up_shape, lo_shape, num_equiv_q, cex_lengths, events = experimental_data

    csv = True

    if not csv: 
        exp_file.write("Experimental Results:\n")
        exp_file.write(f"Number of States: {len(hypothesis[0])}\n")

        exp_file.write(f"Total Number of Preference Queries: {num_pref_q}\n")
        exp_file.write(f"Total Number of Equivalence Queries: {num_equiv_q}\n")
        exp_file.write(f"Total Number of Inequalities: {num_ineq}\n")
        exp_file.write(f"Total Number of Unique Sequences Tested: {num_unique_seq}\n")
        exp_file.write(f"Total Number of Equivalence Classes Generated: {num_ECs}\n")
        exp_file.write(f"Total Number of Unique Table Variables upon Termination: {num_unique_table_vars}\n")
        exp_file.write(f"Upper Table Dimensions upon Termination: {up_shape}\n")
        exp_file.write(f"Lower Table Dimensions upon Termination: {lo_shape}\n")
        exp_file.write(f"CEX Lengths: {cex_lengths}\n\n\n")
    else:
        exp_file.write(f"{len(hypothesis[0])}#{num_pref_q}#{num_equiv_q}#{num_ineq}#{num_ECs}#{num_unique_table_vars}#{num_unique_seq}#{up_shape}#{lo_shape}#{cex_lengths}#{tuple(events)}\n")

def do_modulus_experiment():
    with open(f"mod_exp.csv", "a") as exp_f:
        for sz in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]:
            for _ in range(10):
                modulus_experiment(sz, 200, 3, exp_f)

def optioned_cookie_experiment(exp_file):
    teacher = OptionedCookieEnvironmentTeacher(200)
    hypothesis, experimental_data = symbolic_lstar(teacher.sigma_I, teacher.sigma_O, teacher)
    num_pref_q, num_ineq, num_unique_seq, num_ECs, num_unique_table_vars, up_shape, lo_shape, num_equiv_q, cex_lengths, events = experimental_data

    csv = True

    if not csv: 
        exp_file.write("Experimental Results:\n")
        exp_file.write(f"Number of States: {len(hypothesis[0])}\n")

        exp_file.write(f"Total Number of Preference Queries: {num_pref_q}\n")
        exp_file.write(f"Total Number of Equivalence Queries: {num_equiv_q}\n")
        exp_file.write(f"Total Number of Inequalities: {num_ineq}\n")
        exp_file.write(f"Total Number of Unique Sequences Tested: {num_unique_seq}\n")
        exp_file.write(f"Total Number of Equivalence Classes Generated: {num_ECs}\n")
        exp_file.write(f"Total Number of Unique Table Variables upon Termination: {num_unique_table_vars}\n")
        exp_file.write(f"Upper Table Dimensions upon Termination: {up_shape}\n")
        exp_file.write(f"Lower Table Dimensions upon Termination: {lo_shape}\n")
        exp_file.write(f"CEX Lengths: {cex_lengths}\n")
        exp_file.write(f"Events: {events}\n\n\n")
    else:
        exp_file.write(f"{len(hypothesis[0])}#{num_pref_q}#{num_equiv_q}#{num_ineq}#{num_ECs}#{num_unique_table_vars}#{num_unique_seq}#{up_shape}#{lo_shape}#{cex_lengths}#{tuple(events)}\n")

def do_cookie_experiment():
    with open(f"cookie2.0.csv", "a") as exp_f:
        exp_f.write("'Number of States'#'Num Pref Q'#'Num Equiv Q'#'Num Ineq'#'Num ECs'#'Num Unique Table Vars'#'Num Unique Sequences'#'Upper Dim'#'Lower Dim'#'CEX Lengths'#'Events'\n")
        for _ in range(10):
            optioned_cookie_experiment(exp_f)

def reward_machine_experiment(exp_file, save_file, rm_file, propositions, samples, ids_sampling):
    teacher = RewardMachineTeacher(rm_file, propositions, samples, sophisticated_sampling=ids_sampling)
    hypothesis, experimental_data = symbolic_lstar(teacher.sigma_I, teacher.sigma_O, teacher)
    num_pref_q, num_ineq, num_unique_seq, num_ECs, num_unique_table_vars, up_shape, lo_shape, num_equiv_q, cex_lengths, events = experimental_data
   
    hypothesis_to_rm(hypothesis, teacher, save_file, propositions)
 
    csv = True

    if not csv: 
        exp_file.write("Experimental Results:\n")
        exp_file.write(f"Number of States: {len(hypothesis[0])}\n")

        exp_file.write(f"Total Number of Preference Queries: {num_pref_q}\n")
        exp_file.write(f"Total Number of Equivalence Queries: {num_equiv_q}\n")
        exp_file.write(f"Total Number of Inequalities: {num_ineq}\n")
        exp_file.write(f"Total Number of Unique Sequences Tested: {num_unique_seq}\n")
        exp_file.write(f"Total Number of Equivalence Classes Generated: {num_ECs}\n")
        exp_file.write(f"Total Number of Unique Table Variables upon Termination: {num_unique_table_vars}\n")
        exp_file.write(f"Upper Table Dimensions upon Termination: {up_shape}\n")
        exp_file.write(f"Lower Table Dimensions upon Termination: {lo_shape}\n")
        exp_file.write(f"CEX Lengths: {cex_lengths}\n")
        exp_file.write(f"Events: {events}\n\n\n")
    else:
        exp_file.write(f"{len(hypothesis[0])}#{num_pref_q}#{num_equiv_q}#{num_ineq}#{num_ECs}#{num_unique_table_vars}#{num_unique_seq}#{up_shape}#{lo_shape}#{cex_lengths}#{tuple(events)}\n")

def hypothesis_to_rm(hypothesis, teacher, save_file, propositions):
    states, alpha_I, alpha_O, init_state, delta, output = hypothesis
    moore = Moore(list(states.keys()), alpha_I, alpha_O, delta, init_state, output)
    moore.del_halting_state(value=teacher.halting_value, by_value=True)
    
    mealy = moore.convert_to_mealy()
    ## Now, we need to summarize the Mealy edges. We can do this by inspecting the Mealy or Moore machine
    summarize_transitions(mealy, propositions)
    ## NOTE: We explicitly will not merge terminal states here, because Icarte 2018's RM implementation expects
    ## there to be only a single transition between a pair of states.
    ## Peviously, [Now, we need to ensure there is a single terminal state, so merge all termination states. A termination
    ## state is identified as having no transitions]
    # distinct_termination_state = merge_termination_states(mealy)
    termination_states = find_termination_states(mealy)

    ## Now, we need to convert this Mealy into Reward Machine format
    ## We can record these as output files, and run RL algorithms on them later
    mealy.serialize_to_rm_file(save_file, termination_states)
    #new_rm = RewardMachine("test.rm.txt")
    #print("The new RM")
    #print(new_rm)

def test_rm_conversion(rm_file, propositions):
    rm = RewardMachine(rm_file)

    teacher = RewardMachineTeacher(rm_file, propositions, 200)
    hypothesis, experimental_data = symbolic_lstar(teacher.sigma_I, teacher.sigma_O, teacher)
    num_pref_q, num_ineq, num_unique_seq, num_ECs, num_unique_table_vars, up_shape, lo_shape, num_equiv_q, cex_lengths, events = experimental_data

    ## Now, convert the hypothesis to a Moore machine:
    ## The Hypothesis is in the form 
    ## states, self.sigma_I, self.sigma_O, init_state, delta, output_fnc
    ## States: {row -> prefix}; the state is a the tuple(row)
    ## Delta: { row -> { letter -> row }}
    ## output_fnc: {row -> value}
    states, alpha_I, alpha_O, init_state, delta, output = hypothesis
    print("Original RM:")
    print(rm)
    moore = Moore(list(states.keys()), alpha_I, alpha_O, delta, init_state, output)
    print("Learned Moore with Extra Absorbing State")
    print(moore)
    moore.del_halting_state(value=teacher.halting_value, by_value=True)
    print("Converted Moore to Incomplete with Terminal States")
    print(moore)
    
    mealy = moore.convert_to_mealy()
    print(mealy)
    ## Now, we need to summarize the Mealy edges. We can do this by inspecting the Mealy or Moore machine
    summarize_transitions(mealy, propositions)   
    print("Symbolic Mealy Transitions")
    print(mealy)
    ## Now, we need to ensure there is a single terminal state, so merge all termination states. A termination
    ## state is identified as having no transitions
    distinct_termination_state = merge_termination_states(mealy)
    print("Mealy with merged termination state")
    print(f"Termination State: {distinct_termination_state}")
    print(mealy)

    ## Now, we need to convert this Mealy into Reward Machine format
    ## We can record these as output files, and run RL algorithms on them later
    mealy.serialize_to_rm_file("test.rm.txt", distinct_termination_state)
    new_rm = RewardMachine("test.rm.txt")
    print("The new RM")
    print(new_rm)

def find_termination_states(mealy):
    ## Find all termination states
    termination_set = set()
    for state, transitions in mealy.transitions.items():
        if len(transitions) == 0:
            termination_set.add(state)
    
    ## Remove all termination states from the transition function
    for state in termination_set:
        del mealy.transitions[state]

    return tuple(termination_set)

def merge_termination_states(mealy):
    """
    This function finds all termination states, and merges them into a single state
    """
    ## Find all termination states
    termination_set = set()
    for state, transitions in mealy.transitions.items():
        if len(transitions) == 0:
            termination_set.add(state)

    ## Remove all termination states from the transition function
    for state in termination_set:
        del mealy.transitions[state]

    ## Select one termination state at the representative
    unique_termination_state = termination_set.pop()
    termination_set.add(unique_termination_state)

    ## Now we have only non-terminal states in the transition function
    for state, transitions in mealy.transitions.items():
        for formula, output in transitions.items():
            next_state, value = output
            ## Replace all terminal states with the terminal state representative
            if next_state in termination_set:
                transitions[formula] = tuple((unique_termination_state, value))

    ## Replace all terminal states in the state set with the unique terminal state
    new_states = set(mealy.states) - termination_set
    new_states.add(unique_termination_state)
    mealy.states = tuple(new_states)
    return unique_termination_state
 
def summarize_transitions(mealy, propositions):
    """
    Edges going to the same next state, and with the same output will be collapsed and summarized to the same edge.

    mealy.transitions: {state -> { letter: (state, value)}}
    """
    for source, transitions in mealy.transitions.items():
        ## Transition truth tables for this source state
        transition_truth_tables = dict()
        for letter, output in transitions.items():
            if output not in transition_truth_tables:
                transition_truth_tables[output] = set()
            transition_truth_tables[output].add(letter)
        ## source -> this output has a truth table.
        boolean_formula_transitions = dict()
        for output, truth_table in transition_truth_tables.items():
            ## Convert each truth table to a DNF
            dnf_terms = []
            for true_props in truth_table:
                unused_propositions = set(propositions)
                for prop in true_props:
                    unused_propositions.discard(prop)
                true_portion = sympy.And(*tuple(sympy.Symbol(prop) for prop in true_props))
                false_portion = sympy.And(*tuple((~sympy.Symbol(prop)) for prop in unused_propositions))
                conjunction = sympy.And(true_portion, false_portion)
                dnf_terms.append(conjunction)
            dnf = sympy.Or(*tuple(dnf_terms))
            dnf = sympy.simplify_logic(dnf, form="dnf")
            ## Stringify the DNF into the format that the reward machine likes to take in:
            ## Use ! for not, and make sure there is no white space in the expression
            boolean_formula_transitions[str(dnf).replace("~","!").replace(" ","")] = output
        mealy.transitions[source] = boolean_formula_transitions

def do_reward_machine_experiment(args):
    #samples = 2000
    samples = args.eq_samples
    #A, B = (50, 100)
    A, B = (None, None)
    if args.trial is not None:
        A = args.trial
        B = A + 1
    else:
        A = args.trial_min
        B = args.trial_max
    #task = "t9"
    task = args.task
    #domain = "office" ## office or craft
    domain = args.domain ## office or craft
    prefix = f"lstar_exps/reward_machine_experiments/{domain}-{samples}"

    os.makedirs(prefix, exist_ok=True)

    craft_props = {
        "t1": "ab",   
        "t2": "ac",   
        "t3": "de",   
        "t4": "db",   
        "t5": "afe",  
        "t6": "abcd", 
        "t7": "abcf", 
        "t8": "acf",  
        "t9": "aefg", 
        "t10": "abcfh",
        "t105": "afe",  
        "t106": "abcd", 
        "t107": "abcf", 
        "t108": "acf",  
        "t109": "aefg", 
        "t110": "abcfh",
    }
    office_props = {
        "t1":"fgn",  
        "t2":"egn",  
        "t3":"efgn", 
        "t4":"abcdn",
    }
    domain_props = {
        "office": office_props,
        "craft": craft_props,
    }

    with open(f"{domain}_world_abr_{samples}_{task}.csv.part-{A}-{B}", "a") as exp_f:
        exp_f.write("'Number of States'#'Num Pref Q'#'Num Equiv Q'#'Num Ineq'#'Num ECs'#'Num Unique Table Vars'#'Num Unique Sequences'#'Upper Dim'#'Lower Dim'#'CEX Lengths'#'Events'\n")
        for idx in range(A,B):
            reward_machine_experiment(exp_f, f"{prefix}/{task}.txt.{idx}", f"reward_machines/{domain}/{task}.txt", domain_props[domain][task],samples,args.ids_sampling)
            ## OfficeWorld
            #reward_machine_experiment(exp_f, f"{prefix}/t1.txt.{idx}", "reward_machines/office/t1.txt", "fgn",    samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t2.txt.{idx}", "reward_machines/office/t2.txt", "egn",    samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t3.txt.{idx}", "reward_machines/office/t3.txt", "efgn",   samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t4.txt.{idx}", "reward_machines/office/t4.txt", "abcdn",  samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t0_0.txt.{idx}", "reward_machines/office/t0_0.txt", "ab", samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t0.txt.{idx}", "reward_machines/office/t0.txt", "ab", samples)
           
            ## CraftWorld (MineCraft) 
            #reward_machine_experiment(exp_f, f"{prefix}/{task}.txt.{idx}", f"reward_machines/craft/{task}.txt", craft_props[task],samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t1.txt.{idx}", "reward_machines/craft/t1.txt",         "ab",     samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t2.txt.{idx}", "reward_machines/craft/t2.txt",         "ac",     samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t3.txt.{idx}", "reward_machines/craft/t3.txt",         "de",     samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t4.txt.{idx}", "reward_machines/craft/t4.txt",         "db",     samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t5.txt.{idx}", "reward_machines/craft/t5.txt",         "afe",    samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t6.txt.{idx}", "reward_machines/craft/t6.txt",         "abcd",   samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t7.txt.{idx}", "reward_machines/craft/t7.txt",         "abcf",   samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t8.txt.{idx}", "reward_machines/craft/t8.txt",         "acf",    samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t9.txt.{idx}", "reward_machines/craft/t9.txt",         "aefg",   samples)
            #reward_machine_experiment(exp_f, f"{prefix}/t10.txt.{idx}", "reward_machines/craft/t10.txt",       "abcfh",  samples)

def parse_args():
    parser = argparse.ArgumentParser(description='Provide Filenames')
    parser.add_argument('--eq_samples', type=int,
                        help='Teacher uses <eq_samples> sequence samples during equivalence queries')
    parser.add_argument('--trial', type=int, default=None,
                        help='Trial number')
    parser.add_argument('--trial_min', type=int,
                        help='Minimum trial unit')
    parser.add_argument('--trial_max', type=int,
                        help='Maximum trial unit')
    parser.add_argument('--domain', type=str,
                        help='Domain for learning')
    parser.add_argument('--task', type=str,
                        help='Task for learning')
    parser.add_argument('--ids_sampling', action='store_true', default=False,
                        help='Use IDS Sampling')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    #do_sums_modulus_experiment()
    #do_cookie_experiment()
    #do_simple_regex_experiment()
    do_reward_machine_experiment(args)
    #test_rm_conversion("reward_machines/office/t0_0.txt", "ab")
