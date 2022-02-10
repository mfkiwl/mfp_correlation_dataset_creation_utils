#
# mfp_correlation_data_utils
# Copyright (c) 2022 Miguel Pereira
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Main unit tests to make sure that this package reproduces exactly what was taking
place in the original cloud run instance in terms of creating the dataset.
"""
import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../src/mfp_correlation_data_utils'))
from mfp_correlation_data_utils import nlos_los_data

class TestDatasetReproduced(unittest.TestCase):
    def setUp(self):
        self.folder = "" # link to data here.
        self.data_src = nlos_los_data.data_src(self.folder, 20)

    def test_UCL_data(self):
        UCL_train_data, UCL_train_targets, UCL_test_data, UCL_test_targets = self.data_src.create_ucl_datasets(30)
        UCL_train_data_orig = np.load(self.folder + "/UCL_train_data.npy")
        UCL_train_targets_orig = np.load(self.folder + "/UCL_train_targets.npy")
        UCL_test_data_orig = np.load(self.folder + "/UCL_test_data.npy")
        UCL_test_targets_orig = np.load(self.folder + "/UCL_test_targets.npy")
        self.assertTrue(np.array_equal(UCL_train_data, UCL_train_data_orig) &
                        np.array_equal(UCL_train_targets, UCL_train_targets_orig) &
                        np.array_equal(UCL_test_data, UCL_test_data_orig) &
                        np.array_equal(UCL_test_targets, UCL_test_targets_orig))

    def test_check_UCL_data(self):
        Goodchild_train_data, Goodchild_train_targets, Goodchild_test_data, Goodchild_test_targets = self.data_src.create_goodchild_datasets()
        Goodchild_train_data_orig = np.load(self.folder + "/Goodchild_train_data.npy")
        Goodchild_train_targets_orig = np.load(self.folder + "/Goodchild_train_targets.npy")
        Goodchild_test_data_orig = np.load(self.folder + "/Goodchild_test_data.npy")
        Goodchild_test_targets_orig = np.load(self.folder + "/Goodchild_test_targets.npy")
        self.assertTrue(np.array_equal(Goodchild_train_data, Goodchild_train_data_orig) &
                        np.array_equal(Goodchild_train_targets, Goodchild_train_targets_orig) &
                        np.array_equal(Goodchild_test_data, Goodchild_test_data_orig) &
                        np.array_equal(Goodchild_test_targets, Goodchild_test_targets_orig))
