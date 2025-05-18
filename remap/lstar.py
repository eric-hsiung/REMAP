import itertools
import numpy as np
import z3
import random
from collections import deque

## TEST COVERAGE: [COMPLETE]
class BijectiveIndexMapping:
    def __init__(self):
        self.forward = dict()
        self.backward = dict()
        self.len = 0

    def __str__(self):
        return f"  SIZE = {self.len}\n  ENTRIES: {self.forward}"
    def __repr__(self):
        return f"  SIZE = {self.len}\n  ENTRIES: {self.forward}"

    ## [TESTED][DONE]
    def add(self, el):
        if el not in self.forward:
            sz = len(self.forward)
            self.forward[el] = sz
            self.backward[sz] = el
            self.len = self.len + 1

    ## [TESTED][DONE]
    def __contains__(self, el):
        return el in self.forward

    ## [TESTED][DONE]
    def get(self, el):
        return self.forward[el]

    ## [TESTED][DONE]
    def get_idx(self, idx):
        return self.backward[idx]

    ## [TESTED][DONE]
    def __len__(self):
        return self.len

## TEST COVERAGE: [COMPLETE]
class TableList:
    def __init__(self):
        self.nrows = 0
        self.ncols = 0
        self.table = []

    def __str__(self):
        return f"  SHAPE = ({self.nrows},{self.ncols})\n  Table = {self.__fmt_table__()}"
    
    def __repr__(self):
        return f"  SHAPE = ({self.nrows},{self.ncols})\n  Table = {self.__fmt_table__()}"

    def __fmt_table__(self):
        s = "\n"
        for row in self.table:
            s += f"{row}\n"
        return s
 
    ## [TESTED][DONE]
    def shape(self):
        return (self.nrows, self.ncols)

    ## [TESTED][DONE]
    def resize(self, target_nrows, target_ncols):
        if self.ncols < target_ncols:
            cdiff = target_ncols - self.ncols
            for _ in range(cdiff):
                self.append_col()

        if self.nrows < target_nrows:
            rdiff = target_nrows - self.nrows
            for _ in range(rdiff):
                self.append_row()

    ## [TESTED][DONE]
    def append_row(self, row=None):
        """
        Appends a row to the table. By default, the row is empty
        """
        if row is None:
            row = [None for _ in range(self.ncols)]

        if self.ncols > 0:
            assert len(row) == self.ncols
            self.table.append(list(row))
            self.nrows += 1
        else:
            self.table.append(list(row))
            self.nrows += 1
            self.ncols = len(row)

    ## [TESTED][DONE]
    def append_col(self, col=None):
        """
        Appends a col to the table. By default, the col is empty
        """
        if col is None:
            col = [None for _ in range(self.nrows)]

        if self.nrows > 0:
            assert len(col) == self.nrows
            for idx in range(self.nrows):
                self.table[idx].append(col[idx])
            self.ncols += 1
        else:
            ## Currently there are no rows, so we need to add rows
            for val in col:
                self.table.append([val])
            self.ncols += 1
            self.nrows = len(col)

    ## [TESTED][DONE]
    def get_row(self, idx):
        if idx < self.nrows and idx >= 0:
            return self.table[idx]
        else:
            raise IndexError(f"Bad index: Table has {self.nrows}, but ridx={idx} was requested")

    def num_unique_rows(self):
        s = set()
        for row in self.table:
            s.add(tuple(row))
        return len(s)

    ## [TESTED][DONE]
    def get_entry(self, r, c):
        if r < self.nrows and c < self.ncols and r >= 0 and c >= 0:
            return self.table[r][c]
        else:
            raise IndexError(f"Bad index: Table is {(self.nrows,self.ncols)}, but {(r,c)} was requested")

    ## [TESTED][DONE]
    def set_row(self, idx, row, mk_copy=True):
        if idx < self.nrows and len(row) == self.ncols:
            ## Explicitly make a copy of the row
            if mk_copy:
                self.table[idx] = list(row)
            else:
                self.table[idx] = row
        else:
            raise ValueError("Bad index or bad row values")

    ## [TESTED][DONE]
    def set_entry(self, r, c, v):
        if r < self.nrows and c < self.ncols:
            self.table[r][c] = v
        else:
            raise IndexError(f"Bad index: Table is {(self.nrows,self.ncols)}, but {(r,c)} was requested")

    ## [TESTED][DONE]
    def check_row_equivalence(self, ridx1, ridx2):
        """
        Given a pair of row indices, returns a column index where the entries in the two rows differs
        Otherwise returns -1
        """
        r1 = self.table[ridx1]
        r2 = self.table[ridx2]

        col = -1
        for cidx in range(self.ncols):
            ## Does
            if not r1[cidx].__eq__(r2[cidx]):
                col = cidx
                break
        return ((col == -1), col)

    ## [TESTED][DONE]
    def contains_rows(self, other):
        """
        Check if this table's contains the rows of the other table

        Returns a tuple:
            (True or False, ridx of a row in other.table that is not in self.table)
        """
        this_set = set(tuple(r) for r in self.table)
        nrows, ncols = other.shape()
        absent_row_idx = -1
        for ridx in range(nrows):
            if tuple(other.get_row(ridx)) not in this_set:
                absent_row_idx = ridx
                break
        return ((absent_row_idx == -1), absent_row_idx)

    def get_entry_set(self):
        entry_set = set()
        for row in self.table:
            for entry in row:
                entry_set.add(entry)
        return entry_set

## TEST COVERAGE: [COMPLETE]
class EquivalenceClass:
    def __init__(self, members):
        self.members = members
        self.representative = None
        for e in members:
            self.representative = e
            break

    def update(self, eqclass):
        self.members.update(eqclass.members)
        ## Keep this eqclass's representative the same

    def add(self, member):
        self.members.add(member)

    def repr(self):
        return self.representative

    def __str__(self):
        return f"  SIZE: {len(self.members)}\n  REPRESENTATIVE: {self.representative}\n  MEMBERS: {self.members}\n"

    def __repr__(self):
        return str(self)

## Add utility function for extracting z3 variables from formula

## TEST COVERAGE: [COMPLETE]
## [TESTED][DONE]
def get_vars(f):
    """
    Given a formula, return the leaves in the AST tree
    NOTES:
        f.decl() -- gives the root operator
        f.children() -- list of children
        f.num_args() -- number of children in the list
        f.arg(0), f.arg(1) -- left and right subtrees
    """
    r = set()
    def collect(f):
        ## Base case -- f is a leaf, which means it has no children:
        if f.num_args() == 0:
            r.add(f)
        else: ## f has children
            for c in f.children():
                collect(c)
    collect(f)
    return r

class SymbolicObservationTable:
    """
    This observation table is based on symbolic entries in the table:

    The symbolic observation table is the tuple <S,E,T;C,G>, and it
    consists of the following:
        * S is the set of prefixes
        * E is the set of suffixes
        * T is the sequence-to-output function
        * C is the set of constraints
        * G is the context (set of variables)
    """
    ## [TESTED][DONE]
    def __init__(self, sigma_I, sigma_O, preference_query, equivalence_query,
            prefix = None,
            suffix = None,
            constraints = {"eq": set(), "ineq": set(), "EC": dict(), "repr": dict()},
            context = dict()):
        """
        Prefix set S and suffix set E are each represented by pairs of dictionaries,
        which map sequences to indices, and indices to sequences.
            - .prefix_set: (\Sigma)^* -> row idx
            - .inv_prefix_set: row idx -> (\Sigma)^*
            - .suffix_set: (\Sigma)^* -> col idx
            - .inv_suffix_set: col idx -> (\Sigma)^*

        Context: Maps: Sequence -> Variable
        """
        self.verbose = False
        ## prefixes, suffxies, and prefix-alpha sets
        self.prefix_set = BijectiveIndexMapping()
        self.suffix_set = BijectiveIndexMapping()
        self.prefix_alpha_set = BijectiveIndexMapping()

        ## We represent the upper and lower parts of the table separately, as lists of lists
        self.table_upper = TableList()
        self.table_lower = TableList()
        ## Store a reference to the alphabet
        self.sigma_I = sigma_I
        self.sigma_O = sigma_O
        self.preference_query = preference_query
        self.equivalence_query = equivalence_query

        ## Constraints and Context:
        self.constraints = constraints
        self.context = context

        ## Unique Sequence Tracking
        self.old_entries = set()
        self.old_sorted = deque()

        self.initialize_table(prefix, suffix)

        self.__exp__num_pref_queries = 0
        self.__exp__num_equi_queries = 0
        self.__exp__cex_lengths = []

        self.__exp__events = []
        self.__exp__event_id = -1


    def print(self, s):
        if self.verbose:
            print(s)

    def record_event(self, event_type):
        self.__exp__event_id += 1
        event_id = self.__exp__event_id
        num_states = self.table_upper.num_unique_rows()
       
        num_known_classes = 0 
        RP = self.constraints["repr"]
        for k, v in RP.items():
            if v is not None:
                num_known_classes += 1
        self.__exp__events.append((event_id, event_type, num_states, num_known_classes))

    def record_cex_length(self, cex):
        self.__exp__cex_lengths.append(len(cex))

    def experimental_data(self):
        """
        Returns total number of preference queries that were made
        Total number of unique sequences that were tested
        Total number of equivalence classes
        Number of unique variables in observation table
        Dimensions of upper table
        Dimensions of lower table
        Number of equivalence queries made
        Tuple of the CEX lengths in order that they were received
        Event list
        """
        EC = self.constraints["EC"]
        reps = set()
        for k, v in EC.items():
            reps.add(v.repr())

        return (
            self.__exp__num_pref_queries,
            len(self.constraints["ineq"]),
            len(self.context), 
            len(reps),
            len(self.table_upper.get_entry_set() | self.table_lower.get_entry_set()),
            self.table_upper.shape(),
            self.table_lower.shape(),
            self.__exp__num_equi_queries,
            tuple(self.__exp__cex_lengths),
            self.__exp__events,
        )

    ## [TESTED][DONE]
    def initialize_table(self, prefix, suffix):
        """
        Preconditions: Both tables should be size (0,0).
        Postconditions: After this initialization process,
            1 <= len(prefix set) <= 1 + len(prefix)
            1 <= len(suffix set) <= 1 + len(suffix)
            len(prefix_alpha set) == len(sigma^I)*len(prefix set)
            table_upper.shape() == (len(prefix set) , len(suffix set))
            table_lower.shape() == (len(prefix_alpha set) , len(suffix set))
        """
        ## NOTE: Individual prefixes and individual suffixes are sequences.
        ## Therefore, we require that each element of "prefix" and "suffix" must be a tuple.
        ## Initialize the prefix set
        empty_seq = tuple()
        if prefix is None:
            self.prefix_set.add(empty_seq)
        else:
            self.prefix_set.add(empty_seq)
            for el in prefix:
                if el is None:
                    self.prefix_set.add(empty_seq)
                elif isinstance(el, (tuple,list,str)):
                    self.prefix_set.add(tuple(el))
                else:
                    self.prefix_set.add((el,))

        ## Initialize the suffix set
        if suffix is None:
            self.suffix_set.add(empty_seq)
        else:
            self.suffix_set.add(empty_seq)
            for el in suffix:
                if el is None:
                    self.suffix_set.add(empty_seq)
                elif isinstance(el, (tuple,list,str)):
                    self.suffix_set.add(tuple(el))
                else:
                    self.suffix_set.add((el,))

        ## Initialize the prefix_alpha set based on the intialized prefixes
        for el in self.prefix_set.forward:
            for alpha in self.sigma_I:
                self.prefix_alpha_set.add(concat(el, alpha))

        ## Adjust the table column sizes
        suffix_sz = len(self.suffix_set)
        self.table_upper.ncols = suffix_sz
        self.table_lower.ncols = suffix_sz

        ## Initialize the table entries to None
        for _ in range(len(self.prefix_set)):
            self.table_upper.append_row()
        for _ in range(len(self.prefix_alpha_set)):
            self.table_lower.append_row()

        ## Check Constraints and Context:
        self.print(f"Initial Constraints:\n")
        self.print(self.constraints)
        self.print(f"Initial Context:\n")
        self.print(self.context)

    ## DONE: Add self.constraints["repr"] as a dict() of repr -> value
    ## Need to reflect this in the:
    ## [DONE] Unification function
    ## [DONE] Equivalence Query procedure
    def make_hypothesis(self):
        def is_in(v, it):
            return z3.Or([v == r for r in it])

        self.print("=== Constructing Hypothesis ===\n")        
        self.print(self)

        ## Construct the states
        states = dict()
        init_state = None
        for prefix, ridx in self.prefix_set.forward.items():
            row = tuple(self.table_upper.get_row(ridx))
            if row not in states:
                states[row] = prefix

            if ridx == 0:
                init_state = row

        ## Construct the transition function
        delta = dict()
        for row, prefix in states.items():
            delta[row] = dict()
            for letter in self.sigma_I:
                ridx = self.prefix_alpha_set.get(concat(prefix, letter))
                delta[row][letter] = tuple(self.table_lower.get_row(ridx))

        ## Pass constraints to the SMT solver so that we can get a hypothesis output function:
        solver = z3.Solver()
        solver.reset()
        solver.add(self.constraints["ineq"]) ## Add constraints
        ## Ensure the domain -- though, there might be a more straightforward way to do this..
        for rep, val in self.constraints["repr"].items():
            if val is None:
                solver.add( is_in(rep, self.sigma_O) )
            else:
                solver.add( rep == val)
        sat_unsat = solver.check()
        if sat_unsat == z3.sat:
            model = solver.model()

            ## Construct the output function
            output_fnc = dict()
            for row, prefix in states.items():
                ridx = self.prefix_set.get(prefix)
                var_entry = self.table_upper.get_entry(ridx, 0)
                ## Convert from internal Z3 representation to Python
                output_fnc[row] = model[var_entry].as_long()

            self.__exp__num_equi_queries += 1
            return states, self.sigma_I, self.sigma_O, init_state, delta, output_fnc
        else:
            raise ValueError("SMT Solver returned unsat.")

    def __fmt_equivalence_classes__(self):
        EC = self.constraints["EC"]
        reps = set()
        for k, v in EC.items():
            reps.add(v.repr())

        s = f" --> NUMBER OF EQUIVALENCE CLASSES: {len(reps)}\n"
        counter = 1
        for v in reps:
            s += f"EC{counter}:\n{EC[v]}\n"
            counter+=1

        return s

    def __fmt_repr_values__(self):
        RP = self.constraints["repr"]
        s = f" --> NUMBER OF KNOWN REPR VALUES: {len(RP)}\n"
        s += f"  REPR VALUES:\n"
        for k, v in RP.items():
            s+= f"    {k}=={v}\n"
        return s

    def __fmt_ineqs__(self):
        INEQ = self.constraints["ineq"]
        s = f" --> NUMBER OF INEQUALITIES: {len(INEQ)}\n"
        s += f"  INEQUALITIES:\n"
        for v in INEQ:
            s += f"    {v}\n"
        return s

    def __str__(self):
        """
        Print out everything that we can about the observation table
        """
        s = f"PREFIXES:\n{self.prefix_set}\nPREFIX_ALPHA_SET:\n{self.prefix_alpha_set}\nSUFFIXES:\n{self.suffix_set}\n"
        ## Also double check number of unique variables in the table
        table_entry_set = self.table_upper.get_entry_set() | self.table_lower.get_entry_set()
        s+= f"TABLE ENTRIES:\n"
        s+= f"  --> NUMBER OF UNIQUE ENTRIES: {len(table_entry_set)}\n"
        s+= f"  TABLE ENTRY SET: {table_entry_set}\n"
        s+= f"TABLE_UPPER:\n{self.table_upper}\nTABLE_LOWER:\n{self.table_lower}\n"
        s+= f"EQUIVALENCE_CLASSES: {self.__fmt_equivalence_classes__()}\n"
        s+= f"KNOWN REPRESENTATIVE VALUES: {self.__fmt_repr_values__()}\n"
        s+= f"UNIFIED CONSTRAINTS: {self.__fmt_ineqs__()}\n"
        return s
        

    ## DONE
    ## [TESTED][DONE]
    def is_closed(self):
        """
        The Observation Table is closed iff rowset(table_lower) is a subset of
        rowset(table_upper)

        Returns: a tuple --
            Bool, prefix_alpha

            If Bool is True, prefix_alpha is None
            If Bool is False, prefix_alpha is a tuple
        """
        ## Does the upper table contain all the rows of the lower table?
        status, ridx = self.table_upper.contains_rows(self.table_lower)
        prefix_alpha = None
        ## If not, then find the row index of a row in the lower table that is not
        ## in the upper table, and return it, so that in the next step, we can add that
        ## row to the upper table.
        if not status:
            prefix_alpha = self.prefix_alpha_set.get_idx(ridx)
        return status, prefix_alpha

    ## DONE
    ## [TESTED][DONE]
    def is_consistent(self):
        """
        The purpose of the consistency check: determine whether the transitions
        are deterministic. Remember, the unique rows of S determine the states
        Q.
        Let q \\in Q. If q, a -> q_1, and q, a -> q_2, for q_1 != q_2, then this
        must mean that even though currently q = row(s_1) = row(s_2), row(s_1)
        and row(s_2) must actually be different.
        """
        ## For every pair of identical rows in table_upper:
        row_to_idx = dict()
        nonunique_rows = set()
        for ridx in range(self.table_upper.nrows):
            trow = tuple(self.table_upper.get_row(ridx))
            if trow not in row_to_idx:
                row_to_idx[trow] = [ridx]
            else:
                row_to_idx[trow].append(ridx)
                nonunique_rows.add(trow)

        ## All the rows are unique
        if len(nonunique_rows) == 0:
            return True, None
        ## The rows are not unique -- check all pairs for each row
        for trow in nonunique_rows:
            ## Get pairs
            for s1_idx, s2_idx in itertools.combinations(row_to_idx[trow], 2):
                s1 = self.prefix_set.get_idx(s1_idx)
                s2 = self.prefix_set.get_idx(s2_idx)

                for a in self.sigma_I:
                    s1a = concat(s1, a)
                    s2a = concat(s2, a)
                    r1idx = self.prefix_alpha_set.get(s1a)
                    r2idx = self.prefix_alpha_set.get(s2a)
                    ## Are the two rows symbolically equivalent?
                    status, result = self.table_lower.check_row_equivalence(r1idx, r2idx)
                    if not status:
                        suffix = self.suffix_set.get_idx(result)
                        return status, concat(tuple((a,)), suffix)
        return True, None

    ## DONE
    ## [TESTED][DONE]
    def expand_prefixes(self, prefix_alpha, expand_table=True):
        """
        This is the part of the algorith where we have already identified that
        row(s1*a) is not found in rowspace(S).

        This adds the prefix_alpha in prefix_alpha to the prefix set,
        and is used in the Closed test:

        add the string s.a to S, then extend T to (S u S.A).E
        This means, |S| has increased by 1, and |S.A| has increased by (|A| - 1)

        NOTE: prefix_alpha is a sequence; and we need it to be hashable
        """
        if expand_table:
            ## Add the row from the lower table to the upper table
            ridx_lower = self.prefix_alpha_set.get(prefix_alpha)
            new_row = self.table_lower.get_row(ridx_lower)
            self.table_upper.append_row(new_row)

        ## Add to the prefix set
        self.prefix_set.add(prefix_alpha)
        
        ## Expand the prefix_alpha set
        for alpha in self.sigma_I:
            self.prefix_alpha_set.add(concat(prefix_alpha, alpha))

        ## Return here; a symbolic fill will be called afterwards

    ## DONE
    ## [TESTED][DONE][TESTED in test_is_consistent]
    def expand_suffixes(self, alpha_suffix):
        """
        This expands the alpha_suffix (adds it to the suffix set). This occurs
        in the consistency check
        """
        ## Add a*e to E, and expand the number of columns in the table.
        self.suffix_set.add(alpha_suffix)
        ## Return here; a symbolic fill will be called afterwards

    ## DONE
    ## [TESTED][DONE]
    def symbolic_fill(self):
        ## Make sure to resize the tables
        self.table_upper.resize(len(self.prefix_set), len(self.suffix_set))
        self.table_lower.resize(len(self.prefix_alpha_set), len(self.suffix_set))

        ## TODO: Somehow cache the old entries
        newentries = set()
        #oldentries = set()

        ## Create new fresh variables when required
        for suffix, cidx in self.suffix_set.forward.items():
            for prefix, ridx in self.prefix_set.forward.items():
                ps = concat(prefix, suffix) ## Look up by sequence, since identical sequences should have same output
                if ps not in self.context:
                    sz = len(self.context)
                    self.context[ps] = z3.Int(f"v{sz}")
                    newentries.add(ps)
                #else:
                    ## The entry is either already an old entry, or it is already in the new entry list
                    #oldentries.add(ps)
                self.table_upper.set_entry(ridx, cidx, self.context[ps])
            for prefix, ridx in self.prefix_alpha_set.forward.items():
                ps = concat(prefix, suffix)
                if ps not in self.context:
                    sz = len(self.context)
                    self.context[ps] = z3.Int(f"v{sz}")
                    newentries.add(ps)
                #else:
                    #oldentries.add(ps)
                self.table_lower.set_entry(ridx, cidx, self.context[ps])

        num_pref_queries = 0

        ## Compare the new entries
        sorted_new_entries, new_prefs = self.quicksort(list(newentries))
        num_pref_queries += new_prefs

        ## NOTE: That here, len(sorted_new_entries) <= len(newentries) because we include
        ## only one instance of sequences that are equivalent to one another

        ## Compare the new and old entries
        ## Merge the lists
        merged = deque()
        while len(sorted_new_entries) > 0 and len(self.old_sorted) > 0:
            p1 = sorted_new_entries[-1]
            p2 = self.old_sorted[-1]
            p = self.preference_query(p1, p2)
            self.update_constraint_set(p, p1, p2) ## Send the pair to the contraint set
            num_pref_queries += 1

            if p > 0:
                s = sorted_new_entries.pop()
                merged.appendleft(s)
            elif p < 0:
                s = self.old_sorted.pop()
                merged.appendleft(s)
            else:
                ## They are considered equal, so we prefer taking from old_sorted
                s = self.old_sorted.pop()
                _ = sorted_new_entries.pop()
                merged.appendleft(s)

        while len(sorted_new_entries) > 0:
            s = sorted_new_entries.pop()
            merged.appendleft(s)

        while len(self.old_sorted) > 0:
            s = self.old_sorted.pop()
            merged.appendleft(s)

        self.old_sorted = merged
        
        ## Update the old entries now
        self.old_entries.update(newentries)

        ## Pair combinations -- pairs of (new, new) entries, and pairs of (new, old) entries
        #for p1, p2 in itertools.chain(itertools.combinations(newentries, 2), itertools.product(newentries, oldentries)):
        #    ## Query the teacher
        #    p = self.preference_query(p1, p2) ## Send the seqquences to the preference query
        #    ## Update the constraint set
        #    self.update_constraint_set(p, p1, p2) ## Send the pair to the contraint set
        #    num_pref_queries += 1
        
        self.__exp__num_pref_queries += num_pref_queries

        ## Perform unification
        self.unification()
        ## Return

    def quicksort(self, L):
        num_pref_queries = 0
        if len(L) <= 1:
            return L, num_pref_queries

        ## Pick a random pidx for the pivot
        pidx = random.randint(0,len(L)-1)
        p2 = L[pidx]

        L_list = list()
        R_list = list()

        for idx in range(len(L)):
            if idx == pidx:
                continue
            p1 = L[idx]
            p = self.preference_query(p1, p2)
            self.update_constraint_set(p, p1, p2) ## Send the pair to the contraint set
            num_pref_queries += 1

            if p < 0:
                L_list.append(p1)
            elif p > 0:
                R_list.append(p1)
            else: ## Equal, so don't append anything to either list
                pass ## Creating the equivalence class can be deferred to the unification step

        sorted_L_list, nl_prefs = self.quicksort(L_list)
        sorted_R_list, nr_prefs = self.quicksort(R_list)

        num_pref_queries += nl_prefs
        num_pref_queries += nr_prefs

        sorted_list = list()
        sorted_list.extend(sorted_L_list)
        sorted_list.append(p2)
        sorted_list.extend(sorted_R_list)
        return sorted_list, num_pref_queries

    ## DONE 
    ## [TESTED][DONE]
    def unification(self):
        """
        NOTE: We need an easy way to identify all the equality constraints.

        We will consider each individual constraint as an expression; therefore we will need to be able to identify the
        individual terms in the each expression, and if necessary, modify the expression via substitution.

        We will also need to determine whether an expression contains a particular term.
        """
        ## Step 1: Identify and construct equivalence classes based on the equality constraints,
        ## and establish representatives of each equivalence class.
        ## NOTE: Since we are creating equivalence classes for variables, we don't need to keep
        ## the original equality constraints around (they are encoded into the equivalence classes)
        ## Therefore, we will always just encode the equality constraints into equivalence classes,
        ## and just keep the equivalence classes around. If two variables belong to the same equivalence
        ## class, then they have an implicit equality between them.
        ## NOTE: We also need to take care of the cases where there are no equalities available. In that
        ## case, individual variables go into their own equivalence classes.
        self.print(f"=== >>> UNIFICATION STEP <<< ===\n")
        EC = self.constraints["EC"] ## Dictionary of equivalence classes -- Variable -> EquivalenceClass
        RP = self.constraints["repr"] ## Dictionary of representatives -- Variable -> Value
        self.print(f" PROCESSING EQ: {len(self.constraints['eq'])} equalities to process")
        while len(self.constraints["eq"]) > 0:
            equality = self.constraints["eq"].pop()
            self.print(f"   {equality}")
            ## Parse the equality
            L = equality.arg(0)
            R = equality.arg(1)

            if L in EC and R in EC:
                ## Check if we need to merge equivalence classes
                if not (EC[L] is EC[R]):
                    ## NOTE: We only keep the "left" representative
                    R_rep = EC[R].repr()
                    L_rep = EC[L].repr()
                    ## Add all Right EC class members to the left
                    EC[L].update(EC[R])
                    ## Make all Right EC class members refer to the left EC class
                    R_EC_members = EC[R].members
                    for member in R_EC_members:
                        EC[member] = EC[L]
                    if not bool(L_rep == R_rep):
                        ## Preserve known values:
                        if RP[R_rep] is not None:
                            RP[L_rep] = RP[R_rep]
                        if R_rep in RP:
                            del RP[R_rep] ## If they have different representives, then delete the right one
                            self.print(f"  -  Deleted {R_rep} (via eq)")
                ## Else they already refer to the same equivalence class
            elif L in EC:
                EC[L].add(R)
                EC[R] = EC[L]
            elif R in EC:
                EC[R].add(L)
                EC[L] = EC[R]
            else:
                EC[L] = EquivalenceClass(set((L,R)))
                EC[R] = EC[L]
                rep = EC[L].repr()
                RP[rep] = None
                self.print(f"  +  Added {rep} (via eq)")

            ## NOTE: Now we can do things like check if "EC[a] is EC[b]", which tells us whether
            ## "a" and "b" belong to the same equivalence class or not.
            ## We can also perform substitution by using EC[a].repr() -- this will look up
            ## the representative for the equivalence class that "a" belongs to, and return that
            ## representative for us to use

        ## Step 2: Perform substitution for all terms that are present in our set of inequalities
        self.print(f" PROCESSING INEQ: {len(self.constraints['ineq'])} inequalities to process")
        new_ineqs = set()
        while len(self.constraints["ineq"]) > 0:
            constraint = self.constraints["ineq"].pop()
            self.print(f"    {constraint}")
            ## Parse the constraint to find the variables that are used in it
            variables = get_vars(constraint)
            ## Some variables might not be in an equivalence class, so add them to their own
            for v in variables:
                if v not in EC:
                    EC[v] = EquivalenceClass(set((v,)))
                    rep = EC[v].repr()
                    RP[rep] = None
                    self.print(f"  +  Added {rep} (via ineq)")
            subs = list((v, EC[v].repr()) for v in variables)
            new_constraint = z3.substitute(constraint, subs)
            new_ineqs.add(new_constraint)
        self.constraints["ineq"] = new_ineqs

        ## Step 3: Perform substitution for each entry in the symbolic observation table
        nrows, ncols = self.table_upper.shape()
        self.print(f" PROCESSING UPPER TABLE ENTRIES: SHAPE is ({nrows}, {ncols})")
        for r in range(nrows):
            for c in range(ncols):
                cur_var = self.table_upper.get_entry(r,c)
                sub_var = EC[cur_var].repr()
                self.table_upper.set_entry(r,c, sub_var)
        
        nrows, ncols = self.table_lower.shape()
        self.print(f" PROCESSING LOWER TABLE ENTRIES: SHAPE is ({nrows}, {ncols})")
        for r in range(nrows):
            for c in range(ncols):
                cur_var = self.table_lower.get_entry(r,c)
                sub_var = EC[cur_var].repr()
                self.table_lower.set_entry(r,c, sub_var)

        ## Step 4: Update the context, but this might not be necessary, since we only use the context
        ## for creating fresh variables
        self.print(f" PROCESSING CONTEXT: {len(self.context)} entries")
        for k, v in self.context.items():
            self.context[k] = EC[v].repr()

        self.print(f"UNIQUE TABLE VARIABLES:\n  {self.table_upper.get_entry_set() | self.table_lower.get_entry_set()}")

        ## Step 5: Prune unnecessary constraints, such as representatives not used in the table and constraints that
        ## don't use the table variables.
        

    def set_variable_value(self, seq, val):
        """
        Given a sequence, look up its corresponding variable, and add a constraint stating that the
        variable must be equal to a certain value

        NOTE: Our context should include our sequence
        """
        representative = self.constraints["EC"][self.context[seq]].repr()
        self.constraints["repr"][representative] = val
        self.print(f"SET VARIABLE-> VALUE:\n  SEQ: {seq}\n   VAR: {representative} == VAL {val}\n")


    def update_constraint_set(self, p, p1, p2):
        """
        NOTE: Variables are shared between the upper and lower tables using the context, so each unique
        sequence corresponds to its own unique variable.

        This means all we have to do is look up the corresponding variable in the context using the sequence.
        That way, we don't have to deal with the upper/lower table prefix/suffix indexing.
        """
        if p == 0:
            self.constraints["eq"].add(self.context[p1] == self.context[p2])
        elif p < 0:
            self.constraints["ineq"].add(self.context[p1] < self.context[p2])
        else:
            self.constraints["ineq"].add(self.context[p1] > self.context[p2])

## [TESTED][DONE]
def concat(first, second, as_type=tuple):
    if first is None or second is None:
        if first is None and second is None:
            return as_type()

        if first is None and second is not None:
            if isinstance(second, as_type):
                return second
            else:
                return as_type(second)

        if first is not None and second is None:
            if isinstance(first, as_type):
                return first
            else:
                return as_type(first)

    ## Both are not None
    a = None
    b = None
    if isinstance(first, (tuple, list, str)):
        a = as_type(first)
    else:
        a = as_type((first,))

    if isinstance(second, (tuple, list)):
        b = as_type(second)
    else:
        b = as_type((second,))

    return a + b

def range_prefixes(seq):
    """
    Yields all prefixes of a sequence
    """
    max_len = len(seq)
    for idx in range(max_len):
        yield seq[:idx+1]

def symbolic_lstar(input_alphabet, output_alphabet, teacher):
    """
    Here, we implement the symbolic lstar algorithm, using preferences and I/O examples
    """
    print(f"INITIALIZATION:\n")
    sym_obs_table = SymbolicObservationTable(
        input_alphabet, output_alphabet, teacher.preference_query, teacher.equivalence_query,
        prefix = None, suffix = None,
        constraints = {"eq": set(), "ineq": set(), "EC": dict(), "repr": dict()},
        context = dict())

    
    print(sym_obs_table)

    sym_obs_table.symbolic_fill()
    is_correct = False
    is_consistent = False
    is_closed = False
    hypothesis = None

    sym_obs_table.record_event("initialization")

    while not is_correct:

        is_consistent, alpha_suffix = sym_obs_table.is_consistent()
        is_closed, prefix_alpha = sym_obs_table.is_closed()

        while not (is_closed and is_consistent):
            if not is_consistent:
                sym_obs_table.expand_suffixes(alpha_suffix)
                sym_obs_table.symbolic_fill()
                sym_obs_table.record_event("consistency")

            if not is_closed:
                sym_obs_table.expand_prefixes(prefix_alpha)
                sym_obs_table.symbolic_fill()
                sym_obs_table.record_event("closure")

            #sym_obs_table.symbolic_fill()
            is_consistent, alpha_suffix = sym_obs_table.is_consistent()
            is_closed, prefix_alpha = sym_obs_table.is_closed()

        hypothesis = sym_obs_table.make_hypothesis()
        ## Unpack the hypothesis for the query
        is_correct, result = sym_obs_table.equivalence_query(*hypothesis)
        if not is_correct:
            cex, val = result
            sym_obs_table.record_cex_length(cex)
            for cex_prefix in range_prefixes(cex):
                sym_obs_table.expand_prefixes(cex_prefix, expand_table=False)
            sym_obs_table.symbolic_fill()
            sym_obs_table.set_variable_value(cex, val)
        sym_obs_table.record_event("equivalence")
    return hypothesis, sym_obs_table.experimental_data()
