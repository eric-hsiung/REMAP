import z3
from datastructures import EquivalenceClass

def define_domain(v, it):
    return z3.Or([v == r for r in it])

## Add utility function for extracting z3 variables from formula
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
            if not isinstance(f, (z3.z3.IntNumRef)):
                r.add(f) ## Only add variables; exclude numerical constants
        else: ## f has children
            for c in f.children():
                collect(c)
    collect(f)
    return r

## TEST COVERAGE: [COMPLETE]
## [TESTED][DONE]
def range_prefixes(seq):
    """
    Yields all prefixes of a sequence
    """
    max_len = len(seq)
    for idx in range(max_len):
        yield seq[:idx+1]

def convert_to_equivalence_classes(eq_constraints, EC, RP, print_fnc=print):
    """
    NOTE: This function takes the set of eq_constraints and modifies
          the contents of EC and RP
    """
    print_fnc(f"[PRE EC]")
    for r in RP:
        print_fnc(EC[r])

    while len(eq_constraints) > 0:
        equality = eq_constraints.pop()
        print_fnc(f"   {equality}")
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
                        print_fnc(f"  -  Deleted {R_rep} (via eq)")
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
            print_fnc(f"  +  Added {rep} (via eq)")
    
    print_fnc(f"[POST EC]")
    for r in RP:
        print_fnc(EC[r])
    print_fnc(f"==[POST EC]==")

        ## NOTE: Now we can do things like check if "EC[a] is EC[b]", which tells us whether
        ## "a" and "b" belong to the same equivalence class or not.
        ## We can also perform substitution by using EC[a].repr() -- this will look up
        ## the representative for the equivalence class that "a" belongs to, and return that
        ## representative for us to use
