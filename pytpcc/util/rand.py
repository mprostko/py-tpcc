# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------
# Copyright (C) 2011
# Andy Pavlo
# http://www.cs.brown.edu/~pavlo/
#
# Original Java Version:
# Copyright (C) 2008
# Evan Jones
# Massachusetts Institute of Technology
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# -----------------------------------------------------------------------

from . import nurand
from numba import njit
from numba import prange
import numpy as np
import string

SYLLABLES = [ "BAR", "OUGHT", "ABLE", "PRI", "PRES", "ESE", "ANTI", "CALLY", "ATION", "EING" ]
LOWERCASE_ALPHABET = np.array(list(string.ascii_lowercase))
DIGITS = np.array(list(string.digits))

nurandVar = None # NURand
def setNURand(nu):
    global nurandVar
    nurandVar = nu
## DEF

def NURand(a, x, y):
    """A non-uniform random number, as defined by TPC-C 2.1.6. (page 20)."""
    global nurandVar
    assert x <= y
    if nurandVar is None:
        setNURand(nurand.makeForLoad())

    if a == 255:
        c = nurandVar.cLast
    elif a == 1023:
        c = nurandVar.cId
    elif a == 8191:
        c = nurandVar.orderLineItemId
    else:
        raise Exception("a = " + a + " is not a supported value")

    return (((number(0, a) | number(x, y)) + c) % (y - x + 1)) + x
## DEF

@njit
def number(minimum, maximum):
    assert minimum <= maximum
    if minimum == maximum:
        return minimum
    value = np.random.randint(minimum, maximum + 1)
    assert minimum <= value and value <= maximum
    return value
## DEF

@njit
def numberExcluding(minimum, maximum, excluding):
    """An in the range [minimum, maximum], excluding excluding."""
    assert minimum < maximum
    assert minimum <= excluding and excluding <= maximum

    ## Generate 1 less number than the range
    num = number(minimum, maximum-1)

    ## Adjust the numbers to remove excluding
    if num >= excluding: num += 1
    assert minimum <= num and num <= maximum and num != excluding
    return num
## DEF

@njit
def fixedPoint(decimal_places, minimum, maximum):
    assert decimal_places > 0
    assert minimum < maximum

    multiplier = 1
    for i in range(0, decimal_places):
        multiplier *= 10

    int_min = int(minimum * multiplier + 0.5)
    int_max = int(maximum * multiplier + 0.5)

    return float(number(int_min, int_max) / float(multiplier))
## DEF

def selectUniqueIds(numUnique, minimum, maximum):
    rows = set()
    for i in range(0, numUnique):
        index = None
        while index == None or index in rows:
            index = number(minimum, maximum)
        ## WHILE
        rows.add(index)
    ## FOR
    assert len(rows) == numUnique
    return rows
## DEF

@njit
def astring(minimum_length, maximum_length):
    """A random alphabetic string with length in range [minimum_length, maximum_length]."""
    return ''.join(np.random.choice(LOWERCASE_ALPHABET, size=number(minimum_length, maximum_length)))
## DEF

@njit
def astrings(minimum_length, maximum_length):
    """1D array of random alphabetic strings, their lengths in ranges [[minimum_length], [maximum_length]]."""
    assert len(minimum_length) == len(maximum_length)
    random_strings = [ ]
    for i in range(len(minimum_length)):
        random_strings.append(astring(minimum_length[i], maximum_length[i]))
    assert len(random_strings) == len(minimum_length)
    return random_strings
## DEF

@njit
def nstring(minimum_length, maximum_length):
    """A random numeric string with length in range [minimum_length, maximum_length]."""
    return ''.join(np.random.choice(DIGITS, size=number(minimum_length, maximum_length)))
## DEF

def makeLastName(number):
    """A last name as defined by TPC-C 4.3.2.3. Not actually random."""
    global SYLLABLES
    assert 0 <= number and number <= 999
    indicies = [ number//100, (number//10)%10, number%10 ]
    return "".join(map(lambda x: SYLLABLES[x], indicies))
## DEF

def makeRandomLastName(maxCID):
    """A non-uniform random last name, as defined by TPC-C 4.3.2.3. The name will be limited to maxCID."""
    min_cid = 999
    if (maxCID - 1) < min_cid: min_cid = maxCID - 1
    return makeLastName(NURand(255, 0, min_cid))
## DEF

def shuffle(arg):
    np.random.shuffle(arg)
## DEF
