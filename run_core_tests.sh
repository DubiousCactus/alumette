#! /bin/sh
#
# run_core_tests.sh
# Copyright (C) 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.
#


python -m unittest -v tests/test_tensor.py
python -m unittest -v tests/test_autograd.py
