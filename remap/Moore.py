class Moore:
    def __init__(self, states, input_alphabet, output_alphabet, transitions, initial_state, output_table ):
        """
        states -- list of states
        input_alphabet -- list of input alphabet
        output_alphabet -- list of output alphabet
        transitions -- {state : {input : next_state, output}}
        initial_state -- a state
        output_table -- {state: value}
        """

        self.states = states
        self.input_alphabet = input_alphabet
        self.output_alphabet = output_alphabet
        self.transitions = transitions
        self.output_table = output_table
        self.initial_state = initial_state

    def del_halting_state(self, value=None, by_value=False):
        """
        Removes the halting state by looking up the state with a specific value.
        A halting state is one that has the specified value, and also is absorbing.
        States that transition directly to the halting state are considered terminal states
        """

        def is_absorbing(state):
            return all(v == state for k, v in self.transitions[state].items())

        def all_to_sink(sink, source):
            return all(v == sink for k, v in self.transitions[source].items())

        halt_state = None
        if by_value:
            for k, v in self.output_table.items():
                if v == value and is_absorbing(k):
                    halt_state = k
                    break
            if halt_state is not None:
                ## Remove this state from the output
                del self.output_table[halt_state]
                ## Remove this state from the transitions
                del self.transitions[halt_state]
                ## Terminal states are states which have ALL transitions leading to the HALT state
                ## So identify all the states which have ALL transitions leading to the HALT state
                terminal_states = list()
                for k, v in self.transitions.items():
                    if all_to_sink(halt_state, k):
                        terminal_states.append(k)
                for state in terminal_states:
                    self.transitions[state] = {}
                s = set(self.states)
                s.remove(halt_state)
                self.states = tuple(s)

    def add_halting_state(self, halt_value):
        """
        Adds a single halting (absorbing) state after the terminal state(s)
        """
        halt_added = False
        halt_transition = {a:'HALT' for a in self.input_alphabet}
        for k, v in self.transitions.items():
            if len(v) == 0:
                if not halt_added:
                    self.states.append('HALT')
                    halt_added = True
                self.transitions[k] = halt_transition
        if halt_added:
            self.transitions['HALT'] = {a:'HALT' for a in self.input_alphabet}
            self.output_table['HALT'] = halt_value
            new_output = set(self.output_alphabet)
            new_output.add(halt_value)
            self.output_alphabet = tuple(new_output)

    def convert_to_mealy(self):
        mealy_transitions = {}
        for x in self.transitions.keys():
            try:
                mealy_transitions[x] = {}
                for  a in self.input_alphabet:
                    mealy_transitions[x][a] = (self.transitions[x][a], self.output_table[self.transitions[x][a]])
            except KeyError as e:
                pass

        mealy_input_alphabet = self.input_alphabet
        mealy_output_alphabet = self.output_alphabet
        mealy_initial_state = self.initial_state
        mealy_states = self.states

        from Mealy import Mealy
        mealy_from_moore = Mealy(
            mealy_states,
            mealy_input_alphabet,
            mealy_output_alphabet,
            mealy_transitions,
            mealy_initial_state
        )
        return mealy_from_moore

    def __str__(self):
        output = f"Moore Machine\nStates {self.states}\nInput Alphabet {self.input_alphabet}\nOutput Alphabet {self.output_alphabet}\nTransitions {self.transitions}\nInitial State {self.initial_state}\nOutput Table {self.output_table}\n"

        return output
