#!/bin/python3

"""
Copyright 2019-2020 Paul Yang

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import csv
import json

def add_record(d, k, i):
    """
    append i to d[k], if it did not in d[k]. this create d[k] if it did not exist.
    """
    if k not in d:
        d[k] = [i]
    else:
        if i not in d[k]:
            d[k].append(i)

def get_records(d, k):
    if k in d:
        return list(d[k])
    else:
        return []

class Enumerable(object):
    def __init__(self, data=None):
        if data == None:
            data = []
        if not hasattr(data, "__iter__"):
            raise TypeError(
                "Enumerable must be instantiated with an iterable object")
        self._data = data

    def __iter__(self):
        return iter(self._data)

    @staticmethod
    def identical(x):
        return x

    @staticmethod
    def simple_combine(x, y):
        return (x, y)

    @staticmethod
    def default_group_join_transform(origi, matched):
        return {
                    "self": origi,
                    "matched": matched
                }

    @staticmethod
    def default_group_by_transform(key, values):
        return {
                    "key": key,
                    "values": values
                }

    def join(self, other,
            self_key = identical.__func__,
            other_key = identical.__func__,
            res_transform=simple_combine.__func__,
            is_outer=False):
        """
        All join functions will only pass through the original enumerable once

        @param other: the other itrerable object to be joined, this is the
                      object to be hashed
        @param self_key: generating key for this object
        @param other_key: generating key for the other object
        @param res_transform: a function takes 2 parameters, a record in this
                              object and a record in the other object, and
                              generates a new object as the joining result.

        @remark: if outer join is on and there is no item in other matched with
                 self, the second argument given to res_transform will be None
        """
        return JoinEnmerable(self, other, self_key, other_key,\
                res_transform, is_outer)

    def group_by(self,
            key,
            before_transform = identical.__func__,
            after_transform = default_group_by_transform.__func__):
        return GroupByEnumerable(self, key, before_transform, after_transform)

    def group_join(self, other,
            self_key = identical.__func__,
            other_key = identical.__func__,
            master_transform = identical.__func__,
            hashed_transform = identical.__func__,
            joined_transform = default_group_join_transform.__func__):
        """
        All join functions will only pass through the original enumerable once

        @param other: the other itrerable object to be joined, this is the
                      object to be hashed
        @param self_key: generating key for this object
        @param other_key: generating key for the other object
        @param joined_transform: a function taskes two parameters, the first
                                 one is the selected record, the second one
                                 is an indexable and iterable, which matches
                                 the first. If nothing matches with the first,
                                 then the second should be empty.
                                 The default value will simple combine it to
                                 a dict containing two elements: 'self' and
                                 'matched'
        """
        return GroupJoinEnmerable(self, other, self_key, other_key,\
                master_transform, hashed_transform, joined_transform)

    def to_list(self) -> list:
        return [r for r in self]

    def select(self, transform):
        return SelectEnumerable(self, transform)

    def where(self, cond):
        return WhereEnumerable(self, cond)

    def deDuplication(self, keyer=identical.__func__):
        """
        @param keyer: generating key for this object
        """
        return DeDuplicationEnumerable(self, keyer)

    def accumulate(self, func, ininial=None):
        """
        Applying function on (accumulated, current)
        """
        pass

class JoinEnmerable(Enumerable):
    def __init__(self, data, other,
            self_key = Enumerable.identical,
            other_key = Enumerable.identical,
            res_transform = Enumerable.simple_combine,
            is_outer = False):
        super().__init__(data)
        self._other = other
        self._self_key = self_key
        self._other_key = other_key
        self._res_transform = res_transform
        self._is_outer = is_outer

    def __iter__(self):
        master_hash_table = {}
        for r in self._other:
            add_record(master_hash_table, self._other_key(r), r)

        for self_record in self._data:
            key = self._self_key(self_record)
            if key in master_hash_table and len(master_hash_table[key]) > 0:
                for other_record in master_hash_table[key]:
                    yield self._res_transform(self_record, other_record)
            elif self._is_outer:
                yield self._res_transform(self_record, None)

class GroupJoinEnmerable(Enumerable):

    def __init__(self, data, other,
            self_key = Enumerable.identical,
            other_key = Enumerable.identical,
            master_transform = Enumerable.identical,
            hashed_transform = Enumerable.identical,
            joined_transform = Enumerable.default_group_join_transform):

        super().__init__(data)
        self._other = other
        self._self_key = self_key
        self._other_key = other_key
        self._master_transform = master_transform
        self._hashed_transform = hashed_transform
        self._join_transform = joined_transform

    def __iter__(self):
        master_hash_table = {}
        for r in self._other:
            add_record(
                    master_hash_table,
                    self._other_key(r),
                    self._hashed_transform(r))

        for self_record in self._data:
            key = self._self_key(self_record)

            yield self._join_transform(
                    self._master_transform(self_record),
                    get_records(master_hash_table, key))


class SelectEnumerable(Enumerable):

    def __init__(self, data, transform):

        super().__init__(data)
        self._transform = transform

    def __iter__(self):
        for self_record in self._data:
            yield self._transform(self_record)


class WhereEnumerable(Enumerable):

    def __init__(self, data, cond):
        super().__init__(data)
        self._cond = cond

    def __iter__(self):
        for self_record in self._data:
            if self._cond(self_record):
                yield self_record


class DeDuplicationEnumerable(Enumerable):
    def __init__(self, data, keyer=Enumerable.identical):
        super().__init__(data)
        self._keyer = keyer

    def __iter__(self):
        used = set()
        for r in self._data:
            key = self._keyer(r)
            if r not in used:
                used.add(key)
                yield r


if __name__ == '__main__':
    def transform(x, y):
        if y is not None:
            return {**x, **y}
        else:
            return x

    with open('test/exam.csv', 'r') as fexam, \
            open('test/student.csv', 'r') as fstudent:
        exams = csv.DictReader(fexam)
        students = csv.DictReader(fstudent)
        res = Enumerable(students).join(exams,
                self_key = lambda x : x['Id'],
                other_key = lambda x : x['Id'],
                res_transform = lambda x, y : {**x, **y},
                is_outer = False).to_list()
        print(json.dumps(res, indent=4))

    with open('test/exam.csv', 'r') as fexam, \
            open('test/student.csv', 'r') as fstudent:
        exams = csv.DictReader(fexam)
        students = csv.DictReader(fstudent)
        res = Enumerable(students).join(exams,
                self_key = lambda x : x['Id'],
                other_key = lambda x : x['Id'],
                res_transform = transform,
                is_outer = True).to_list()
        print(json.dumps(res, indent=4))

    with open('test/exam.csv', 'r') as fexam, \
            open('test/student.csv', 'r') as fstudent:
        exams = csv.DictReader(fexam)
        students = csv.DictReader(fstudent)
        res = Enumerable(students).group_join(exams,
                self_key = lambda x : x['Id'],
                other_key = lambda x : x['Id']
                ).to_list()
        print(json.dumps(res, indent=4))

class GroupByEnumerable(Enumerable):

    def __init__(self, data,
            key = Enumerable.identical,
            before_transform = Enumerable.identical,
            after_transform = Enumerable.default_group_by_transform):

        super().__init__(data)
        self._key = key
        self._before_transform = before_transform
        self._after_transform = after_transform

    def __iter__(self):
        master_hash_table = {}
        for r in self._data:
            add_record(
                    master_hash_table,
                    self._key(r),
                    self._before_transform(r))

        for key, values in master_hash_table.items():
            yield self._after_transform(key, values)


