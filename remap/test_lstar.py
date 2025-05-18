import itertools
import random
import pytest
import numpy as np
from lstar import BijectiveIndexMapping, TableList, EquivalenceClass, SymbolicObservationTable
from lstar import get_vars, concat, symbolic_lstar
from rm_problem import ModulusEnvironment
import z3

class TestSymbolicObservationTable:
    @pytest.mark.parametrize("alphabet,prefix,prefix_alpha,suffix,upper_shape,lower_shape", [
        [
            ("a","b"),
            None,
            ("a","b"),
            None,
            (1,1),
            (2,1),
        ],
        [
            ("a","b"),
            ("",),
            ("a","b"),
            ("","a","b"),
            (1,3),
            (2,3),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b"),
            (3,3),
            (6,3),
        ],
        [
            (1,2,3,4),
            None,
            (1,2,3,4),
            None,
            (1,1),
            (4,1),
        ],
        [
            (1,2,3,4),
            (None, 1,2,3,4),
            ((1,),(2,),(3,),(4,), (1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4)),
            (None, 1,2,3,4),
            (5,5),
            (20,5),
        ],
        [
            (('',), ('a',), ('b',), ('ab',)),
            (tuple(), ('',), ('a',), ('b',), ('ab',)),
            (   ('',), ('a',), ('b',), ('ab',),
                ('',''), ('', 'a'), ('','b'), ('','ab'),
                ('a',''), ('a', 'a'), ('a','b'), ('a','ab'),
                ('b',''), ('b', 'a'), ('b','b'), ('b','ab'),
                ('ab',''), ('ab', 'a'), ('ab','b'), ('ab','ab'),
            ),
            None,
            (5,1),
            (20,1),
        ],
    ]) 
    def test_table_initialization(self, alphabet, prefix, prefix_alpha, suffix, upper_shape, lower_shape):
        def process_list(a_list):
            return all(map(lambda x: x is None, a_list))

        M = SymbolicObservationTable(alphabet, None, None, None, prefix = prefix, suffix = suffix)
        prefix_len = 0
        suffix_len = 0
        if prefix is None:
            prefix_len = 1
        else:
            prefix_len = len(prefix)

        if suffix is None:
            suffix_len = 1
        else:
            suffix_len = len(suffix)

        assert 1 <= len(M.prefix_set) and len(M.prefix_set) <= 1 + prefix_len
        assert 1 <= len(M.suffix_set) and len(M.suffix_set) <= 1 + suffix_len
        assert len(M.prefix_alpha_set) == len(alphabet) * len(M.prefix_set)
        assert M.table_upper.shape() == upper_shape
        assert M.table_lower.shape() == lower_shape
        ## All elements of the table should be None
        assert all(map(process_list, M.table_upper.table))

    @pytest.mark.parametrize("alphabet,prefix,prefix_alpha,suffix,upper,lower,closed", [
        [
            ("a","b"),
            None,
            ("a","b"),
            None,
            (
                ("v1",),
            ),
            (
                ("v1",),
                ("v2",),
            ),
            False
        ],
        [
            ("a","b"),
            None,
            ("a","b"),
            None,
            (
                ("v1",),
            ),
            (
                ("v1",),
                ("v1",),
            ),
            True
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b"),
            (
                ("v1","v2","v3"),
                ("v1","v5","v6"),
                ("v4","v1","v3"),
            ),
            (
                ("v1","v2","v3"),
                ("v1","v5","v6"),
                ("v4","v1","v3"),
                ("v4","v1","v3"),
                ("v1","v2","v3"),
                ("v4","v1","v3"),
            ),
            True
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b"),
            (
                ("v1","v2","v3"),
                ("v1","v5","v6"),
                ("v4","v1","v3"),
            ),
            (
                ("v1","v2","v3"),
                ("v1","v5","v6"),
                ("v4","v1","v3"),
                ("v4","v2","v3"),
                ("v4","v1","v3"),
                ("v4","v2","v3"),
            ),
            False
        ],
    ]) 
    def test_is_closed(self, alphabet, prefix, prefix_alpha, suffix, upper, lower, closed):
        M = SymbolicObservationTable(alphabet, None, None, None, prefix = prefix, suffix = suffix)
        for idx, row in enumerate(upper):
            M.table_upper.set_row(idx, tuple(z3.Int(v) for v in row))
        for idx, row in enumerate(lower):
            M.table_lower.set_row(idx, tuple(z3.Int(v) for v in row))

        result, sa = M.is_closed()
        print("Prefixes")
        print(M.prefix_set)
        print("Prefix Alpha")
        print(M.prefix_alpha_set)
        print("Suffixes")
        print(M.suffix_set)
        print("Upper")
        print(M.table_upper)
        print("Lower")
        print(M.table_lower)

        assert result == closed
    
    @pytest.mark.parametrize("alphabet,prefix,prefix_alpha,suffix,upper,lower,is_consistent,sig_suf,new_suffix", [
        [
            ("a","b"),
            None,
            ("a","b"),
            None,
            (
                ("v1",),
            ),
            (
                ("v1",),
                ("v2",),
            ),
            True,
            None,
            (tuple(),),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            None,
            (
                ("v1",),
                ("v2",),
                ("v1",),
            ),
            (
                ("v2",),
                ("v1",),
                ("v3",),
                ("v4",),
                ("v2",),
                ("v1",),
            ),
            True,
            None,
            (tuple(),),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            None,
            (
                ("v1",),
                ("v2",),
                ("v1",),
            ),
            (
                ("v2",),
                ("v1",),
                ("v3",),
                ("v4",),
                ("v3",),
                ("v1",),
            ),
            False,
            ("a",tuple()),
            (tuple(),tuple("a"))
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b"),
            (
                ("v1","v2","v3"),
                ("v1","v5","v6"),
                ("v1","v2","v3"),
            ),
            (
                ("v1","v5","v6"),
                ("v1","v2","v3"),
                ("v4","v1","v3"),
                ("v5","v1","v3"),
                ("v1","v5","v6"),
                ("v1","v2","v3"),
            ),
            True,
            None,
            (tuple(""),tuple("a"),tuple("b")),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b"),
            (
                ("v1","v2","v3"),
                ("v1","v5","v6"),
                ("v1","v2","v3"),
            ),
            (
                ("v1","v5","v6"),
                ("v1","v2","v3"),
                ("v4","v1","v3"),
                ("v5","v1","v3"),
                ("v1","v2","v3"),
                ("v1","v2","v3"),
            ),
            False,
            ("a","a"),
            (tuple(""),tuple("a"),tuple("b"),tuple("aa")),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b","aa"),
            (
                ("v1","v2","v1","v4"),
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
            ),
            (
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
                ("v4","v4","v4","v4"),
                ("v3","v2","v1","v3"),
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
            ),
            True,
            None,
            tuple(tuple(el) for el in ("","a","b","aa")),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b","aa"),
            (
                ("v1","v2","v1","v4"),
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
            ),
            (
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
                ("v4","v4","v4","v4"),
                ("v3","v2","v1","v3"),
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v1"),
            ),
            False,
            (tuple("b"),tuple("aa")),
            tuple(tuple(el) for el in ("","a","b","aa","baa")),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b","aa"),
            (
                ("v1","v2","v1","v4"),
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
            ),
            (
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
                ("v4","v4","v4","v4"),
                ("v3","v2","v1","v3"),
                ("v2","v4","v1","v4"),
                ("v1","v2","v1","v4"),
            ),
            False,
            (tuple("a"),tuple("b")),
            tuple(tuple(el) for el in ("","a","b","aa","ab")),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b","aa"),
            (
                ("v1","v2","v1","v4"),
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
            ),
            (
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
                ("v4","v4","v4","v4"),
                ("v3","v2","v1","v3"),
                ("v2","v4","v3","v3"),
                ("v1","v2","v1","v4"),
            ),
            False,
            (tuple("a"),tuple("aa")),
            tuple(tuple(el) for el in ("","a","b","aa","aaa")),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b","aa"),
            (
                ("v1","v2","v1","v4"),
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
            ),
            (
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
                ("v4","v4","v4","v4"),
                ("v3","v2","v1","v3"),
                ("v2","v4","v3","v4"),
                ("v1","v2","v4","v4"),
            ),
            False,
            (tuple("b"),tuple("b")),
            tuple(tuple(el) for el in ("","a","b","aa","bb")),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b","aa"),
            (
                ("v1","v2","v1","v4"),
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
            ),
            (
                ("v2","v4","v3","v4"),
                ("v1","v2","v1","v4"),
                ("v4","v4","v4","v4"),
                ("v3","v2","v1","v3"),
                ("v2","v4","v3","v4"),
                ("v1","v1","v1","v4"),
            ),
            False,
            (tuple("b"),tuple("a")),
            tuple(tuple(el) for el in ("","a","b","aa","ba")),
        ],
    ]) 
    def test_is_consistent(self, alphabet, prefix, prefix_alpha, suffix, upper, lower, is_consistent, sig_suf, new_suffix):
        M = SymbolicObservationTable(alphabet, None, None, None, prefix = prefix, suffix = suffix)
        for idx, row in enumerate(upper):
            M.table_upper.set_row(idx, tuple(z3.Int(v) for v in row))
        for idx, row in enumerate(lower):
            M.table_lower.set_row(idx, tuple(z3.Int(v) for v in row))

        result, sigma_suffix = M.is_consistent()
        print("Prefixes")
        print(M.prefix_set)
        print("Prefix Alpha")
        print(M.prefix_alpha_set)
        print("Suffixes")
        print(M.suffix_set)
        print("Upper")
        print(M.table_upper)
        print("Lower")
        print(M.table_lower)
        print(f"Sigma + Suffix: {sigma_suffix}")
        assert result == is_consistent
        if sig_suf is None:
            assert sig_suf == sigma_suffix
        else:
            assert concat(*sig_suf) == sigma_suffix
        if not result:
            M.expand_suffixes(sigma_suffix)
        for s in new_suffix:
            assert s in M.suffix_set

        for s in M.suffix_set.forward:
            assert s in new_suffix

    @pytest.mark.parametrize("alphabet,prefix,prefix_alpha,suffix,upper,lower,pa_list,new_upper_list,closed_list", [
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b"),
            (
                ("v1","v2","v3"),
                ("v1","v5","v6"),
                ("v1","v2","v3"),
            ),
            (
                ("v1","v5","v6"),
                ("v1","v2","v3"),
                ("v4","v1","v3"),
                ("v5","v1","v3"),
                ("v1","v2","v3"),
                ("v1","v2","v3"),
            ),
            (tuple("aa"), tuple("ab")),
            (
                (
                    ("v1","v2","v3"),
                    ("v1","v5","v6"),
                    ("v1","v2","v3"),
                    ("v4","v1","v3"),
                ),
                (
                    ("v1","v2","v3"),
                    ("v1","v5","v6"),
                    ("v1","v2","v3"),
                    ("v4","v1","v3"),
                    ("v5","v1","v3"),
                ),
            ),
            (False, True),
        ],
        [
            ("a","b"),
            ("", "a", "b"),
            ("a","b","aa", "ab", "ba", "bb"),
            ("","a","b"),
            (
                ("v1","v2","v3"),
                ("v1","v5","v6"),
                ("v1","v2","v3"),
            ),
            (
                ("v1","v5","v6"),
                ("v1","v2","v3"),
                ("v1","v2","v3"),
                ("v1","v2","v3"),
                ("v4","v1","v3"),
                ("v5","v1","v3"),
            ),
            (tuple("ba"), tuple("bb")),
            (
                (
                    ("v1","v2","v3"),
                    ("v1","v5","v6"),
                    ("v1","v2","v3"),
                    ("v4","v1","v3"),
                ),
                (
                    ("v1","v2","v3"),
                    ("v1","v5","v6"),
                    ("v1","v2","v3"),
                    ("v4","v1","v3"),
                    ("v5","v1","v3"),
                ),
            ),
            (False, True),
        ],
    ])
    def test_expand_prefixes(self, alphabet, prefix, prefix_alpha, suffix, upper, lower, pa_list, new_upper_list, closed_list):
        M = SymbolicObservationTable(alphabet, None, None, None, prefix = prefix, suffix = suffix)
        for idx, row in enumerate(upper):
            M.table_upper.set_row(idx, tuple(z3.Int(v) for v in row))
        for idx, row in enumerate(lower):
            M.table_lower.set_row(idx, tuple(z3.Int(v) for v in row))

        result, prefix_sigma = M.is_closed()
        assert result == False
        for pa, new_upper, closed in zip(pa_list, new_upper_list, closed_list):
            assert prefix_sigma == pa

            M.expand_prefixes(prefix_sigma,expand_table = True)
            nrows, ncols = M.table_upper.shape()
            assert nrows == len(new_upper)
            assert ncols == len(suffix)

            for idx, r in enumerate(new_upper):
                assert tuple(z3.Int(v) for v in r) == tuple(M.table_upper.get_row(idx))
            assert prefix_sigma in M.prefix_set.forward
            assert prefix_sigma in M.prefix_alpha_set.forward
            for a in alphabet:
                assert concat(prefix_sigma, a) in M.prefix_alpha_set.forward
            result, prefix_sigma = M.is_closed()
            assert result == closed

    @pytest.mark.parametrize("alphabet,prefix,suffix,eqs,ineqs,ecs,pre_upper,pre_lower,pre_auto_context", [
        [
            ("a","b"),
            ("","a","b"),
            None,
            (
                ("ve","va"),
                ("va","vb"),
                ("ve","vb"),
                ("vaa","vbb"),
                ("vab","vba"),
            ),
            (
                ("vb","vba",1),
                ("vbb","vba",1),
                ("vb","vbb",1),
            ),
            (
                ("ve","va","vb"),
                ("vaa","vbb"),
                ("vab","vba"),
            ),
            (
                ("ve",),
                ("va",),
                ("vb",),
            ),
            (
                ("va",),
                ("vb",),
                ("vaa",),
                ("vab",),
                ("vba",),
                ("vbb",),
            ),
            ("","a","b","aa","ab","ba","bb"),
        ],
        [
            ("a","b"),
            ("","a","b"),
            None,
            tuple(),
            (
                ("ve", "va", 1),
                ("vb","vba",1),
                ("vbb","vba",1),
                ("vb","vbb",1),
                ("va", "vaa", 1),
                ("va", "vab",1),
            ),
            (
                ("ve",),
                ("va",),
                ("vb",),
                ("vaa",),
                ("vab",),
                ("vba",),
                ("vbb",),
            ),
            (
                ("ve",),
                ("va",),
                ("vb",),
            ),
            (
                ("va",),
                ("vb",),
                ("vaa",),
                ("vab",),
                ("vba",),
                ("vbb",),
            ),
            ("","a","b","aa","ab","ba","bb"),
        ],
        [
            ("1","2","3"),
            None,
            None,
            (
                ("v1", "v2"),
                ("v3", "ve"),
                ("v1", "v3"),
                ("ve", "v1"),
            ),
            tuple(),
            (
                ("v1","v2","v3","ve"),
            ),
            (
                ("ve",),
            ),
            (
                ("v1",),
                ("v2",),
                ("v3",),
            ),
            ("","1","2","3"),
        ]

    ])
    def test_unification(self, alphabet, prefix, suffix, eqs, ineqs, ecs, pre_upper, pre_lower, pre_auto_context):
        ## Initialize the observation table, constraints, and context
        constraints = {
            "eq": set( z3.Int(L) == z3.Int(R) for L, R in eqs),
            "ineq": set( z3.Int(L) > z3.Int(R) if P > 0 else z3.Int(L) < z3.Int(R) for L, R, P in ineqs),
            "EC": dict(),
            "repr": dict()
        }
        variables = {tuple(s):z3.Int(f"v{s}") for s in pre_auto_context if len(s) > 0}
        variables[tuple()] = z3.Int("ve")
        M = SymbolicObservationTable(alphabet, None, None, None, prefix = prefix, suffix = suffix, constraints = constraints, context = variables)
        for idx, row in enumerate(pre_upper):
            M.table_upper.set_row(idx, tuple(z3.Int(v) for v in row))
        for idx, row in enumerate(pre_lower):
            M.table_lower.set_row(idx, tuple(z3.Int(v) for v in row))

        ## Run the unification algorithm
        M.unification()

        ## Post conditions
        ## All equalities have been converted to equivalence classes
        assert len(M.constraints["eq"]) == 0 ## Equalities have been converted to equivalence classes

        for equiv_class in ecs:
            equiv_class_set = set(z3.Int(el) for el in equiv_class)
            representatives = set()
            for member in equiv_class_set:
                assert M.constraints["EC"][member].members == equiv_class_set
                rep = M.constraints["EC"][member].repr()
                representatives.add(rep)
                assert rep in equiv_class_set
            ## Each equivalence class should have only a single representative
            assert len(representatives) == 1

        ## Ensure we have only the specified number of equivalence classes
        num_classes = 0
        unique_sets = set()
        for k, v in M.constraints["EC"].items():
            unique_sets.add(v.repr())

        assert len(unique_sets) == len(ecs)

        ## Ensure that each variable in the context is a representative
        for k, v in M.context.items():
            assert M.context[k] == M.constraints["EC"][v].repr()

        ## Ensure that each variable in the observation table is a representative
        nrows, ncols = M.table_upper.shape()
        for r in range(nrows):
            for c in range(ncols):
                cur_var = M.table_upper.get_entry(r,c)
                assert cur_var == M.constraints["EC"][cur_var].repr()
        
        nrows, ncols = M.table_lower.shape()
        for r in range(nrows):
            for c in range(ncols):
                cur_var = M.table_lower.get_entry(r,c)
                assert cur_var == M.constraints["EC"][cur_var].repr()

        ## Ensure that each constraint is now written only in terms of representatives
        for ineq in M.constraints["ineq"]:
            for v in get_vars(ineq):
                assert v == M.constraints["EC"][v].repr()

        ## We should have at most the same number of constraints as in ineqs
        assert len(M.constraints["ineq"]) <= len(ineqs)

    @pytest.mark.parametrize("alphabet,prefix,suffix,dataset", [
        [
            ("a","b"),
            ("","a","b"),
            ("",),
            (
                ("", 7),
                ("a", 7),
                ("b", 7),
                ("aa", 5),
                ("ab", 3),
                ("ba", 3),
                ("bb", 5),
            ),
        ],
    ])
    def test_symbolic_fill(self, alphabet, prefix, suffix, dataset):
        """
        Tests the symbolic fill process
        """
        seq2val = {tuple(k):v for k, v in dataset}
        def prefQ(s1, s2):
            if seq2val[s1] == seq2val[s2]:
                return 0
            elif seq2val[s1] > seq2val[s2]:
                return 1
            else:
                return -1
        M = SymbolicObservationTable(alphabet, None, prefQ, None, prefix = prefix, suffix = suffix)
        M.symbolic_fill()
        ## NOTE: This also performs unification internally.

        assert M.table_upper.shape() == (len(prefix), len(suffix))
        assert M.table_lower.shape() == (len(prefix)*len(alphabet), len(suffix))
        assert len(M.context) == len(dataset)

        ## Count the expected number of equivalence classes that should exist
        unique_outputs = set()
        for k, v in seq2val.items():
            unique_outputs.add(v)

        assert len(M.constraints["EC"]) == len(dataset)
        representatives = set()
        for k, v in M.constraints["EC"].items():
            representatives.add(v.repr())

        assert len(unique_outputs) == len(representatives)

        ## The table should be fill only with representatives
        nrows, ncols = M.table_upper.shape()
        for r in range(nrows):
            for c in range(ncols):
                cur_var = M.table_upper.get_entry(r,c)
                assert cur_var == M.constraints["EC"][cur_var].repr()
        
        nrows, ncols = M.table_lower.shape()
        for r in range(nrows):
            for c in range(ncols):
                cur_var = M.table_lower.get_entry(r,c)
                assert cur_var == M.constraints["EC"][cur_var].repr()
        
        ## Ensure that each constraint is now written only in terms of representatives
        for ineq in M.constraints["ineq"]:
            for v in get_vars(ineq):
                assert v == M.constraints["EC"][v].repr()

    def test_symbolic_lstar(self):
        class ModulusTeacher:
            def __init__(self, modulus_size, seq_sample_size):
                self.env = ModulusEnvironment(modulus_size, deterministic=True)
                self.q0 = self.env.sample_initial_state() ## Select an initial state
                self.sigma_O = tuple(v for v in range(modulus_size))
                self.sigma_I = self.env.actions
                self.rng = np.random.default_rng()
                self.seq_sample_size = seq_sample_size

            def sample_sequences(self, quantity):
                """
                Uses a geometric distribution to sample a sequence.
                """
                p = 0.2
                seq_lengths = self.rng.geometric(p, quantity) - 1
                sequences = []
                for length in seq_lengths:
                    ## Generate a random sequence of the desired length
                    if length == 0:
                        sequences.append(tuple())
                    else:
                        sequences.append(tuple(random.choices(self.sigma_I, k=length)))
                return sequences

            def evaluate_sequence(self, s):
                q = self.q0

                for a in s:
                    q = self.env.sample_transition(q, a)

                return q

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
                """
                def evaluate_hypothesis(seq):
                    q = init_state
                    print(delta)
                    for a in seq:
                        q = delta[q][a]
                    return output_fnc[q]

                sequences = self.sample_sequences(self.seq_sample_size)

                for seq in sequences:
                    teacher_output = self.evaluate_sequence(seq)
                    learner_output = evaluate_hypothesis(seq)

                    if teacher_output != learner_output:
                        return False, (seq, teacher_output)

                return True, None

        teacher = ModulusTeacher(7, 10)
        symbolic_lstar(tuple(teacher.sigma_I), teacher.sigma_O, teacher)
        #symbolic_lstar(tuple( (el,) for el in teacher.sigma_I), teacher.sigma_O, teacher)

