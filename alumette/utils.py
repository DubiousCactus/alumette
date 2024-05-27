#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

from typing import List

import numpy as np


def allclose(*tensors, **kwargs) -> bool:
    return np.allclose(
        *[t.data for t in tensors],
        atol=kwargs.get("atol", 1e-5),
        rtol=kwargs.get("rtol", 1e-5)
    )
