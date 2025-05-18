from reward_machines.reward_machine import RewardMachine
from reward_machines.reward_machine_utils import evaluate_dnf
from reward_machine_mod import RewardMachine as RewardMachineMod
from Mealy import Mealy
from Moore import Moore
import itertools
import random
import numpy as np
from search import AutomataProblem
from experiment import RewardMachineTeacher
import argparse

def powerset(s):
    ## Powerset recipe is from https://docs.python.org/3/library/itertools.html#itertools-recipes
    return tuple("".join(x) for x in itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))

def rm_to_complete_mealy(rm, base_alphabet):
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
    alphabet = powerset(base_alphabet)
    output_alphabet = set()
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

def check_accuracy_and_isomorphism(ground_truth_rm, learned_rm, propositions):
    """
    Given a ground truth RM and a learned RM, check accuracy and isomorphism
    """
    quantity = 5000
    geom_param = 0.2
    seq_list = generate_test_sequences(ground_truth_rm, learned_rm, quantity, geom_param, propositions)

    ##  Compute accuracy first:
    num_correct = 0
    num_incorrect = 0
    cex = None
    for sequence in seq_list:
        ground_val = ground_truth_rm.eval_sequence(sequence)
        learned_val = learned_rm.eval_sequence(sequence)
        num_correct += int(ground_val == learned_val)
        if cex is None and ground_val != learned_val:
            cex = sequence

    accuracy = num_correct / len(seq_list)
    
    return accuracy, num_correct, len(seq_list), ground_truth_rm.count_states(), ground_truth_rm.count_transitions(), learned_rm.count_states(), learned_rm.count_transitions(), cex

def run_isomorphism_tests(args):
    """
    args.eq_samples ## number of EQ samples
    args.trial ## trial number
    args.domain ## office or craft
    args.task ## task id (int)
    """
    ## Office World:
    office_propositions = {
        1:"fgn",
        2:"egn",
        3:"efgn",
        4:"abcdn",
    }
    ## Craft World:
    craft_propositions = {
        1:"ab",
        2:"ac",
        3:"de",
        4:"db",
        5:"afe",
        6:"abcd",
        7:"abcf",
        8:"acf",
        9:"aefg",
        10:"abcfh",
        105:"afe",
        106:"abcd",
        107:"abcf",
        108:"acf",
        109:"aefg",
        110:"abcfh",
    }
    props = dict()
    props["office"] = office_propositions
    props["craft"] = craft_propositions

    save_dir = f"lstar_exps/reward_machine_experiments/isomorphism/{args.domain}"
    equiv_samples = args.eq_samples
    ## NOTE, number 10 is not done yet.
    rm_file = f"reward_machines/{args.domain}/t{args.task}.txt"
    ground_truth_rm = RewardMachineMod(rm_file)
    propositions = props[args.domain][args.task]
    with open(f"{save_dir}/t{args.task}.csv.{equiv_samples}.trial.{args.trial}", "a") as f_exp:
        f_exp.write("'Accuracy'#'Num Correct'#'Total Tested'#'Num GRM States'#'Num GRM Transitions'#'Num LRM States'#'Num LRM Transitions'#'Counterexample'\n")
        learned_rm_file = f"lstar_exps/reward_machine_experiments/{args.domain}-{equiv_samples}/t{args.task}.txt.{args.trial}"
        learned_rm = RewardMachineMod(learned_rm_file)
        print(f"Processing {learned_rm_file} .....")
        r = check_accuracy_and_isomorphism(ground_truth_rm, learned_rm, propositions)
        f_exp.write(f"{r[0]}#{r[1]}#{r[2]}#{r[3]}#{r[4]}#{r[5]}#{r[6]}#{r[7]}\n")

def generate_test_sequences(ground_truth_rm, learned_rm, quantity, geom_param, propositions):
    """
    Generate quantity random sequences, where each sequence has a length drawn from a geometric distribution.
    
    Additionally, generate quantity random positive sequences via sampling allowable paths in the ground_truth_rm

    All generated sequences have a length of at least 1
    """
    ## Geometric distribution sequences
    alphabet = powerset(propositions)
    p = geom_param

    rng = np.random.default_rng()
    seq_lengths = rng.geometric(p, quantity)
    geometric_sequences = []
    for length in seq_lengths:
        ## Generate a random sequence of the desired length
        for _ in range(7):
            geometric_sequences.append(tuple(random.choices(alphabet, k=length)))
    ## Ground truth RM sequences
    num_states = ground_truth_rm.count_states()
    ## First, generate all possible paths of length at most num_states
    ap = AutomataProblem(ground_truth_rm)
    all_solutions = ap.iterative_deepening_search(num_states)
    cached_options = dict() ## Formula to list of propositions satisfying the formula

    positive_sampled_sequences = list()

    for solution in all_solutions:
        for _ in range((len(solution)-1)*len(alphabet)*500):
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
    return geometric_sequences

def test_ground_truth_rm_sampling(ground_truth_rm, propositions):
    ## Ground truth RM sequences
    num_states = ground_truth_rm.count_states()
    ## First, generate all possible paths of length at most num_states
    ap = AutomataProblem(ground_truth_rm)
    all_solutions = ap.iterative_deepening_search(num_states)
    ## Convert the solutions to sequences of propositions
    alphabet = powerset(propositions)

    cached_options = dict() ## Formula to list of propositions satisfying the formula

    positive_sampled_sequences = list()

    for solution in all_solutions:
        for _ in range(23):
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
    return positive_sampled_sequences

def parse_args():
    """
    args.eq_samples ## number of EQ samples
    args.trial ## trial number
    args.domain ## office or craft
    args.task ## task id (int)
    """
    parser = argparse.ArgumentParser(description='Provide Filenames')
    parser.add_argument('--eq_samples', type=int,
                        help='Teacher uses <eq_samples> sequence samples during equivalence queries')
    parser.add_argument('--trial', type=int, default=None,
                        help='Trial number')
    parser.add_argument('--domain', type=str,
                        help='Domain for learning')
    parser.add_argument('--task', type=int,
                        help='Task for learning')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    run_isomorphism_tests(args)
    #rm_file = "reward_machines/office/t1.txt"
    #rm = RewardMachine(rm_file)
    #rm = RewardMachineMod(rm_file)
    #print(rm)
    #test_ground_truth_rm_sampling(rm, "fgn")
    exit()


    ## Here, the RM is a Mealy machine, with state and output
    ## transitions defined separately:
    ## State Transitions: {state: {state: input}}
    ## Output Transitions: {state: {state: output}}

    ## In Mealy, the accepted format is:
    ## transitions -- {state : {input : (state, output)}}
    transitions, alphabet, output_alphabet = rm_to_complete_mealy(rm, "fgn")
    print(transitions)
    print(alphabet)
    print(output_alphabet)
    states = list(rm.U)
    states.append(rm.terminal_u)
    mealy = Mealy(states, alphabet, output_alphabet, transitions, rm.u0)
    print(mealy)
    #moore_states, moore_input_alphabet, moore_output_alphabet, moore_transitions, moore_output_table, moore_initial_state = mealy.convert_to_moore()
    #moore = Moore(moore_states, moore_input_alphabet, moore_output_alphabet, moore_transitions, moore_initial_state, moore_output_table)
    #print(moore)
 
    moore, more_output = mealy.to_moore()
    #moore_states, moore_input_alphabet, moore_output_alphabet, moore_transitions, moore_output_table, moore_initial_state, more_output = mealy.to_moore()
    #moore = Moore(moore_states, moore_input_alphabet, moore_output_alphabet, moore_transitions, moore_initial_state, moore_output_table)
    print(moore)
    print(more_output)
    mealy2 = moore.convert_to_mealy()
    print(mealy2)
