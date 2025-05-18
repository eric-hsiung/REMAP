import itertools
import random
import pytest
import numpy as np
from datastructures import BijectiveIndexMapping, TableList, EquivalenceClass
import z3

class TestBijectiveIndexMapping:
    @pytest.mark.parametrize("list_of_elems,length", [
        [
            ("a","b","c","d"),
            4,
        ],
    ])
    def test_add(self, list_of_elems, length):
        M = BijectiveIndexMapping()
        assert len(M) == 0
        for el in list_of_elems:
            M.add(el)

        assert len(M) == length

        for el in list_of_elems:
            assert M.get_idx(M.get(el)) == el

        for el in list_of_elems:
            assert el in M

        for el in M.forward:
            assert el in list_of_elems

class TestTableList:

    @pytest.mark.parametrize("rc1,rc2,rc3", [
        [
            (1,1),
            (1,2),
            (2,2),
        ],
        [
            (1,1),
            (2,1),
            (2,2),
        ],
        [
            (1,4),
            (1,5),
            (5,9),
        ],
        [
            (1,1),
            (5,1),
            (10,10),
        ],
    ])
    def test_resize(self, rc1, rc2, rc3):
        M = TableList()
        assert M.shape() == (0,0)
        M.resize(*rc1)
        assert M.shape() == rc1
        M.resize(*rc2)
        assert M.shape() == rc2
        M.resize(*rc3)
        assert M.shape() == rc3

    def test_construct_from_empty(self):
        M = TableList()
        assert M.shape() == (0,0)
        M.append_row(row=None)
        assert M.shape() == (1,0)
        M.append_col(col=None)
        assert M.shape() == (1,1)
        
        M = TableList()
        assert M.shape() == (0,0)
        M.append_col(col=None)
        assert M.shape() == (0,1)
        M.append_row(row=None)
        assert M.shape() == (1,1)
        
        M = TableList()
        assert M.shape() == (0,0)
        M.append_col(col=None)
        M.append_col(col=None)
        M.append_col(col=None)
        M.append_col(col=None)
        assert M.shape() == (0,4)
        M.append_row(row=None)
        assert M.shape() == (1,4)
        M.append_row(row=None)
        assert M.shape() == (2,4)

        M = TableList()
        assert M.shape() == (0,0)
        M.append_row(row=None)
        M.append_row(row=None)
        M.append_row(row=None)
        M.append_row(row=None)
        assert M.shape() == (4,0)
        M.append_col(col=None)
        assert M.shape() == (4,1)

        M = TableList()
        assert M.shape() == (0,0)
        M.append_row(row=[1,2,3,4])
        assert M.shape() == (1,4)
        
        M = TableList()
        assert M.shape() == (0,0)
        M.append_col(col=[1,2,3,4])
        assert M.shape() == (4,1)

    def test_row_col_access(self):
        M = TableList()
        assert M.shape() == (0,0)
        M.append_row(row=[1,2,3,4])
        assert M.shape() == (1,4)

        with pytest.raises(IndexError) as exinfo:
            row = M.get_row(2)
        assert "Bad index" in str(exinfo)

        with pytest.raises(IndexError) as exinfo:
            row = M.get_row(-1)
        assert "Bad index" in str(exinfo)

        with pytest.raises(IndexError) as exinfo:
            row = M.get_row(1)
        assert "Bad index" in str(exinfo)
        
        row = M.get_row(0)
        assert tuple(row) == tuple([1,2,3,4])
       
        new_row = [1,2,3,4]
        new_row.reverse()
        M.set_row(0, new_row)
        row = M.get_row(0)
        assert tuple(row) == tuple([4,3,2,1])
        assert not(M.get_row(0) is new_row)
        
        M.set_row(0, new_row, mk_copy=False)
        row = M.get_row(0)
        assert tuple(row) == tuple([4,3,2,1])
        assert M.get_row(0) is new_row
        for idx, val in enumerate([4,3,2,1]):
            assert M.get_entry(0,idx) == val

        M.set_entry(0,0, -1)
        assert M.get_entry(0,0) == -1


    @pytest.mark.parametrize("table,equiv_rows,neq_rows", [
        [
            (
                (1,2,3,4),
                (1,2,3,4),
            ),
            (0, 1),
            tuple(),
        ],
        [
            (
                (1,2,3,4),
                (1,2,3,4),
                (1,2,5,4),
            ),
            (0, 1),
            (2,),
        ],
        [
            (
                (1,2,3,4),
                (1,7,3,4),
                (1,2,5,4),
            ),
            tuple(),
            (0,1,2),
        ],
        [
            (
                (1,2,3,4),
                (1,7,3,4),
                (1,2,5,4),
                (1,2,3,4),
            ),
            (0,3),
            (1,2),
        ],
    ])
    def test_check_row_equivalence(self, table, equiv_rows, neq_rows):
        M = TableList()
        for row in table:
            M.append_row(row=row)

        for r1, r2 in itertools.combinations(equiv_rows, 2):
            assert (True, -1) == M.check_row_equivalence(r1,r2)
            assert (True, -1) == M.check_row_equivalence(r2,r1)

        for r1, r2 in itertools.chain(itertools.combinations(neq_rows, 2), itertools.product(neq_rows, equiv_rows)):
            assert (True, -1) != M.check_row_equivalence(r1,r2)
            assert (True, -1) != M.check_row_equivalence(r2,r1)

    @pytest.mark.parametrize("table,dataset,constructor", [
        [
            (
                (1,2,3,4),
                (1,2,3,4),
            ),
            (
                ((0, 1), True, -1),
            ),
            None,
        ],
        [
            (
                (1,2,5,4),
                (1,2,3,4),
            ),
            (
                ((0, 1), False, 2),
            ),
            None,
        ],
        [
            (
                (1,2,3,4),
                (1,7,3,4),
                (1,2,5,4),
                (1,2,3,4),
            ),
            (
                ((0, 1), False, 1),
                ((0, 2), False, 2),
                ((0, 3), True, -1),
                ((1, 2), False, 1),
                ((1, 3), False, 1),
                ((2, 3), False, 2),
            ),
            None,
        ],
        [
            (
                ("v1","v2","v3","v4"),
                ("v1","v7","v3","v4"),
                ("v1","v2","v5","v4"),
                ("v1","v2","v3","v4"),
            ),
            (
                ((0, 1), False, 1),
                ((0, 2), False, 2),
                ((0, 3), True, -1),
                ((1, 2), False, 1),
                ((1, 3), False, 1),
                ((2, 3), False, 2),
            ),
            z3.Int,
        ],
    ])
    def test_check_row_equivalence_col(self, table, dataset, constructor):
        M = TableList()
        if constructor is None:
            for row in table:
                M.append_row(row=row)
        else:
            for row in table:
                M.append_row(row=[constructor(el) for el in row])

        for indices, result, col in dataset:
            r1, r2 = indices
            assert (result, col) == M.check_row_equivalence(r1,r2)
            assert (result, col) == M.check_row_equivalence(r2,r1)

    @pytest.mark.parametrize("t1,t2,absent_indices,constructor", [
        [
            (
                (1,2,3,4),
                (1,2,3,4),
            ),
            (
                (1,2,3,4),
                (1,2,3,4),
            ),
            set(),
            None,
        ],
        [
            (
                (1,2,3,4),
                (1,2,3,4),
            ),
            (
                (1,2,3,4),
                (4,4,4,4),
            ),
            set((1,)),
            None,
        ],
        [
            (
                (1,2,3,4),
                (1,2,3,4),
            ),
            (
                (4,4,4,4),
                (4,2,3,4),
            ),
            set((0,1,)),
            None,
        ],
        [
            (
                ("v1","v2","v3","v4"),
                ("v1","v7","v3","v4"),
                ("v1","v2","v5","v4"),
                ("v1","v2","v3","v4"),
            ),
            (
                ("v1","v2","v3","v4"),
                ("v1","v9","v3","v4"),
                ("v1","v2","v5","v4"),
                ("v1","v2","v3","v4"),
            ),
            set((1,)),
            z3.Int,
        ],
        [
            (
                ("v1","v2","v3","v4"),
                ("v1","v7","v3","v4"),
                ("v1","v2","v5","v4"),
                ("v1","v2","v3","v4"),
            ),
            (
                ("v1","v2","v3","v4"),
                ("v1","v9","v3","v4"),
                ("v1","v2","v9","v4"),
                ("v1","v2","v3","v4"),
            ),
            set((1,2)),
            z3.Int,
        ],
    ])
    def test_contains_rows(self, t1, t2, absent_indices, constructor):
        M1 = self.construct_table(t1, constructor)
        M2 = self.construct_table(t2, constructor)

        result, ridx = M1.contains_rows(M2)

        if len(absent_indices) == 0:
            assert (result, ridx) == (True, -1)
        else:
            assert result == False
            assert ridx in absent_indices


    def construct_table(self, table, constructor):
        M = TableList()
        if constructor is None:
            for row in table:
                M.append_row(row=row)
        else:
            for row in table:
                M.append_row(row=[constructor(el) for el in row])
        return M

class TestEquivalenceClass:

    def test_equiv_class(self):
        s = {1,2,3}
        b = {3,4,5}
        EC = EquivalenceClass(s)
        ED = EquivalenceClass(b)
        r = EC.repr()
        q = ED.repr()
        assert r in s
        assert q in b

        EC.update(ED)
        assert EC.repr() == r
        assert EC.members == s | b

        s = set(z3.Int(el) for el in {"v1","v2","v3"})
        b = set(z3.Int(el) for el in {"v3","v4","v5"})
        EC = EquivalenceClass(s)
        ED = EquivalenceClass(b)
        r = EC.repr()
        q = ED.repr()
        assert r in s
        assert q in b

        EC.update(ED)
        assert EC.repr() == r
        assert EC.members == s | b
