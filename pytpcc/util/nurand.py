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

from numba import njit
from numba import int8, int16
from numba.experimental import jitclass
import numpy as np

@njit
def makeForLoad():
    """Create random NURand constants, appropriate for loading the database."""
    cLast = np.random.randint(0, 256)
    cId = np.random.randint(0, 1024)
    orderLineItemId = np.random.randint(0, 8192)
    return NURandC(cLast, cId, orderLineItemId)

@njit
def validCRun(cRun, cLoad):
    """Returns true if the cRun value is valid for running. See TPC-C 2.1.6.1 (page 20)"""
    cDelta = abs(cRun - cLoad)
    return 65 <= cDelta and cDelta <= 119 and cDelta != 96 and cDelta != 112

@njit
def makeForRun(loadC):
    """Create random NURand constants for running TPC-C. TPC-C 2.1.6.1. (page 20) specifies the valid range for these constants."""
    cRun = np.random.randint(0, 256)
    while validCRun(cRun, loadC.cLast) == False:
        cRun = np.random.randint(0, 256)
    assert validCRun(cRun, loadC.cLast)

    cId = np.random.randint(0, 1024)
    orderLineItemId = np.random.randint(0, 8192)
    return NURandC(cRun, cId, orderLineItemId)

spec = [
    ('cLast', int16),
    ('cId', int16),
    ('orderLineItemId', int16)
]

@jitclass(spec)
class NURandC:
    def __init__(self, cLast, cId, orderLineItemId):
        self.cLast = cLast
        self.cId = cId
        self.orderLineItemId = orderLineItemId
