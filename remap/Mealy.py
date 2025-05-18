from Moore import Moore
class Mealy:

    def __init__(self, states, input_alphabet, output_alphabet, transitions, initial_state):
        """
        states -- list of states
        input_alphabet -- list of input alphabet
        output_alphabet -- list of output alphabet
        transitions -- {state : {input : (next_state, output)}}
        initial_state -- a state
        """
        self.states = states
        self.input_alphabet = input_alphabet
        self.output_alphabet = output_alphabet
        self.transitions = transitions
        self.initial_state = initial_state

    def serialize_to_rm_file(self, filepath, terminal_states):
        """
        We serialize the Mealy machine into RM format for further evaluation.
        First, we map states -> natual numbers
        """
        state_map = {state:idx for idx, state in enumerate(self.states)}

        termination_state_list = list(state_map[ts] for ts in terminal_states)

        with open(filepath, "w") as f:
            ## First specify initial and terminal states
            f.write(f"{state_map[self.initial_state]} # initial state\n")
            f.write(f"{termination_state_list} # terminal state")
            ## Specify the transitions
            for state, transitions in self.transitions.items():
                A = state_map[state]
                for formula, output in transitions.items():
                    next_state, value = output
                    B = state_map[next_state]
                    f.write(f"\n({A},{B},'{formula}',ConstantRewardFunction({value}))")

    def to_moore(self):
        """
        Converts a Mealy machine to a Moore machine
            
        Remember, our Mealy transition table is of the form
        transitions -- {state : {input : (next_state, output)}}
        """
        ## Step 1. Find all next_states in the transition table which have
        ## multiple outputs associated with them. That is, for each next state, determine
        ## if the edges coming into the next_state have more than one output value.
        unused_state = len(self.states)

        ## Initialization
        incoming_outputs = {state:set() for state in self.states}

        for state, edges in self.transitions.items():
            for alpha, value in edges.items():
                next_state, output = value
                incoming_outputs[next_state].add(output)

        moore_transitions = dict() ## {state: {alpha: next_state}}
        moore_outputs = dict() ## {state: output}
        split_merge = dict() ## {old_state: [new_state, new_state, new_state]}
        output_less_states = list()
        
        ## Step 2. For all states that had more than one input, perform state splitting.
        for state, in_values in incoming_outputs.items():
            num_partitions = len(in_values)
            if num_partitions > 1:
                split_merge[state] = [state]
                moore_outputs[state] = in_values.pop()
                moore_transitions[state] = dict()
                for _ in range(num_partitions - 1):
                    split_merge[state].append(unused_state)
                    moore_outputs[unused_state] = in_values.pop()
                    moore_transitions[unused_state] = dict()
                    unused_state += 1
            elif num_partitions == 1:
                split_merge[state] = [state]
                moore_outputs[state] = in_values.pop()
                moore_transitions[state] = dict()
            else: ## num_partitions == 0
                split_merge[state] = [state]
                moore_outputs[state] = None
                moore_transitions[state] = dict()
                output_less_states.append(state)

        ## Step 3. Construct the transition table
        ## At this point, we know each state's output value already.
        ## So, now we just need to populate the transitions properly
        for state, edges in self.transitions.items():
            cur_was_split = len(split_merge[state]) > 1
            for alpha, value in edges.items():
                next_state, output = value
                ## Need to consider the following cases:
                ## A. state size = 1, next_state size = 1
                ## B. state size = 1, next_state size > 1 ( next_state was split )
                ## C. state size > 1, next_state size = 1 ( state was split )
                ## D. state size > 1, next_state size > 1 ( both state and next_state were split )
                nex_was_split = len(split_merge[next_state]) > 1
                if (not cur_was_split) and (not nex_was_split):
                    moore_transitions[state][alpha] = next_state
                elif (not cur_was_split) and nex_was_split:
                    ## Need to look up which state to transition to
                    for split_state in split_merge[next_state]:
                        if output == moore_outputs[split_state]:
                            moore_transitions[state][alpha] = split_state
                            break
                elif cur_was_split and (not nex_was_split):
                    ## All states in the split state should transition to the same state
                    for split_state in split_merge[state]:
                        moore_transitions[split_state][alpha] = next_state
                else: ## Both cur and nex were split
                    for split_next_state in split_merge[next_state]:
                        if output == moore_outputs[split_next_state]:
                            for split_state in split_merge[state]:
                                moore_transitions[split_state][alpha] = split_next_state
                            break

        ## Step 4. Handle the initial state. If the initial state had no input, then it currently has no input
        requires_additional_output = False
        if len(output_less_states) > 0:
            requires_additional_output = True

        ## Step 5. Make sure we pick the original initial state, or one of its split clones
        moore_initial_state = split_merge[self.initial_state][0]
        moore_states = list(moore_transitions.keys())
        
        moore = Moore(moore_states, self.input_alphabet, self.output_alphabet, moore_transitions, moore_initial_state, moore_outputs)
        return moore, requires_additional_output

    def __str__(self):
        output = f"Mealy Machine\nStates: {self.states}\nTransitions: {self.transitions}\nInitial State: {self.initial_state}\nInput Alphabet: {self.input_alphabet}\nOutput Alphabet: {self.output_alphabet}\n"

        return output
