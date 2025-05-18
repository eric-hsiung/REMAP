import copy
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

    def difference(self, other):
        """
        Returns items in self that are not in other.
        """
        options = self.forward.keys() - other.forward.keys()
        return options

## TEST COVERAGE: [COMPLETE]
class TableList:
    def __init__(self):
        self.nrows = 0
        self.ncols = 0
        self.table = []

    def clone(self):
        """
        Creates an entirely independent clone of the table
        """
        cloned = TableList()
        cloned.resize(self.nrows, self.ncols)
        for idx in range(self.nrows):
            cloned.set_row(idx, self.get_row(idx), mk_copy=True)
        return cloned

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

    def __len__(self):
        return len(self.members)

    def update(self, eqclass):
        self.members.update(eqclass.members)
        ## Keep this eqclass's representative the same

    def make_pairs(self):
        """
        Given the members list, construct pairs with full coverage.
        """
        if len(self.members) <= 1:
            return list()

        L = list(self.members)
        N = len(L)
        return [L[i] == L[i+1] for i in range(N-1)] 

    def add(self, member):
        self.members.add(member)

    def repr(self):
        return self.representative

    def __str__(self):
        return f"  SIZE: {len(self.members)}\n  REPRESENTATIVE: {self.representative}\n  MEMBERS: {self.members}\n"

    def __repr__(self):
        return str(self)

    def clone(self):
        return EquivalenceClass(copy.copy(self.members))
