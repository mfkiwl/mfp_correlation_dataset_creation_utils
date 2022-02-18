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
from os import path, listdir
import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import StratifiedShuffleSplit


class data_src():
    """
    Class that gets initialised with a data source, where datasource refers
    to the base file location that contains all of the NLOS and LOS recorded
    data used in the paper:

    - base_folder:
        - UCL_Circ1_Final_Corr_Feature_Results
            - NLOS
                - 2021-07-14_10_14_45_Channel_1_GPS_PRN10.csv   (corr. data)
            - LOS
                - 2021-07-14_10_14_45_Channel_1_GPS_PRN8.csv    (corr. data)
            - UCL_Labelled_SatteliteRay_Data_circuit_1.csv      (ground truths)

        - UCL_Circ2_Final_Corr_Feature_Results
            - NLOS
            - LOS
            - UCL_Labelled_SatteliteRay_Data_circuit_2.csv

        - UCL_Circ3_Final_Corr_Feature_Results
            - NLOS
            - LOS
            - UCL_Labelled_SatteliteRay_Data_circuit_3.csv


        - Goodchild_Final_Corr_Feature_Results
            - NLOS
            - LOS
            - Goodchild_Labelled_SatelliteRay_Data.csv
    """

    cn0_elevation_coeffs = [-1.13241278e-05, 2.05615841e-03, -1.31254923e-01,
                            3.48043259e+00, -1.78140207e+00]
    train_test_split = 0.2
    rand_seed = 10

    def __init__(self, base_folder: str, S_num: int):
        # set the base folder (critical)
        self.base_folder = base_folder
        # set the serialisation number when the correlation outputs are joined
        self.S_num = S_num

        # get the LOS and NLOS files for the two (actually 4) datasets.
        self.__UCL1_path = path.join(
            base_folder, "UCL_Circ1_Final_Corr_Feature_Results")
        self.__UCL1_LOS_files = [
            f for f in listdir(
                path.join(
                    self.__UCL1_path,
                    "LOS")) if "correlation_output.csv" in f]
        self.__UCL1_NLOS_files = [
            f for f in listdir(
                path.join(
                    self.__UCL1_path,
                    "NLOS")) if "correlation_output.csv" in f]

        self.__UCL2_path = path.join(
            base_folder, "UCL_Circ2_Final_Corr_Feature_Results")
        self.__UCL2_LOS_files = [
            f for f in listdir(
                path.join(
                    self.__UCL2_path,
                    "LOS")) if "correlation_output.csv" in f]
        self.__UCL2_NLOS_files = [
            f for f in listdir(
                path.join(
                    self.__UCL2_path,
                    "NLOS")) if "correlation_output.csv" in f]

        self.__UCL3_path = path.join(
            base_folder, "UCL_Circ3_Final_Corr_Feature_Results")
        self.__UCL3_LOS_files = [
            f for f in listdir(
                path.join(
                    self.__UCL3_path,
                    "LOS")) if "correlation_output.csv" in f]
        self.__UCL3_NLOS_files = [
            f for f in listdir(
                path.join(
                    self.__UCL3_path,
                    "NLOS")) if "correlation_output.csv" in f]

        self.__Goodchild_path = path.join(
            base_folder, "Goodchild_Final_Corr_Feature_Results")
        self.__Goodchild_LOS_files = [
            f for f in listdir(
                path.join(
                    self.__Goodchild_path,
                    "LOS")) if "correlation_output.csv" in f]
        self.__Goodchild_NLOS_files = [
            f for f in listdir(
                path.join(
                    self.__Goodchild_path,
                    "NLOS")) if "correlation_output.csv" in f]

        # get the ground truth files
        self.__UCL1_gts = path.join(
            self.__UCL1_path,
            "UCL_Labelled_SatteliteRay_Data_circuit_1.csv")
        self.__UCL2_gts = path.join(
            self.__UCL2_path,
            "UCL_Labelled_SatteliteRay_Data_circuit_2.csv")
        self.__UCL3_gts = path.join(
            self.__UCL3_path,
            "UCL_Labelled_SatteliteRay_Data_circuit_3.csv")
        self.__Goodchild_gts = path.join(
            self.__Goodchild_path,
            "Goodchild_Labelled_SatelliteRay_Data.csv")

    # define the interface
    def create_ucl_datasets(self, lines_per_file: int):
        """
        Returns the train and test UCL dataset as a tuple (train, train_targets, test, test_targets).
        """
        self.lines_per_UCL_file = lines_per_file

        UCL1_LOS_data = self.__create_dataset(
            self.__UCL1_path,
            "LOS",
            self.__UCL1_LOS_files,
            self.__UCL1_gts,
            self.S_num,
            num_rows=lines_per_file)
        UCL2_LOS_data = self.__create_dataset(
            self.__UCL2_path,
            "LOS",
            self.__UCL2_LOS_files,
            self.__UCL2_gts,
            self.S_num,
            num_rows=lines_per_file)
        UCL3_LOS_data = self.__create_dataset(
            self.__UCL3_path,
            "LOS",
            self.__UCL3_LOS_files,
            self.__UCL3_gts,
            self.S_num,
            num_rows=lines_per_file)

        UCL1_NLOS_data = self.__create_dataset(
            self.__UCL1_path,
            "NLOS",
            self.__UCL1_NLOS_files,
            self.__UCL1_gts,
            self.S_num,
            num_rows=lines_per_file)
        UCL2_NLOS_data = self.__create_dataset(
            self.__UCL2_path,
            "NLOS",
            self.__UCL2_NLOS_files,
            self.__UCL2_gts,
            self.S_num,
            num_rows=lines_per_file)
        UCL3_NLOS_data = self.__create_dataset(
            self.__UCL3_path,
            "NLOS",
            self.__UCL3_NLOS_files,
            self.__UCL3_gts,
            self.S_num,
            num_rows=lines_per_file)

        # Combine the UCL data
        UCL_LOS_data = np.vstack([UCL1_LOS_data, UCL2_LOS_data, UCL3_LOS_data])
        UCL_NLOS_data = np.vstack(
            [UCL1_NLOS_data, UCL2_NLOS_data, UCL3_NLOS_data])
        # now stack the NLOS and LOS, together with their targets
        UCL_Combined_Serialised_Data = np.vstack((UCL_NLOS_data, UCL_LOS_data))
        # create the corresponding targets
        UCL_Combined_Serialised_Targets = np.vstack((np.zeros((len(UCL_NLOS_data), 1), dtype=np.int),
                                                     np.ones((len(UCL_LOS_data), 1), dtype=np.int)))

        # now split the data using sklearn stratified shuffle split
        shuffler = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.train_test_split,
            random_state=self.rand_seed).split(
            UCL_Combined_Serialised_Data,
            UCL_Combined_Serialised_Targets)
        indices = [(train_idx, validation_idx)
                   for train_idx, validation_idx in shuffler][0]
        UCL_train_data = UCL_Combined_Serialised_Data[indices[0]]
        UCL_train_targets = UCL_Combined_Serialised_Targets[indices[0]]
        UCL_test_data = UCL_Combined_Serialised_Data[indices[1]]
        UCL_test_targets = UCL_Combined_Serialised_Targets[indices[1]]

        return UCL_train_data, UCL_train_targets, UCL_test_data, UCL_test_targets

    def create_goodchild_datasets(self, lines_per_file: int = -1):
        """
        Returns the train and test Goodchild dataset as a tuple.
        """
        self.lines_per_Goodchild_file = lines_per_file

        Goodchild_LOS_data = self.__create_dataset(
            self.__Goodchild_path,
            "LOS",
            self.__Goodchild_LOS_files,
            self.__Goodchild_gts,
            self.S_num,
            num_rows=lines_per_file)
        Goodchild_NLOS_data = self.__create_dataset(
            self.__Goodchild_path,
            "NLOS",
            self.__Goodchild_NLOS_files,
            self.__Goodchild_gts,
            self.S_num,
            num_rows=lines_per_file)
        # combine the data into a single 2D array
        Goodchild_Combined_Serialised_Data = np.vstack(
            (Goodchild_NLOS_data, Goodchild_LOS_data))
        Goodchild_Combined_Serialised_Targets = np.vstack((np.zeros((len(Goodchild_NLOS_data), 1), dtype=np.int),
                                                           np.ones((len(Goodchild_LOS_data), 1), dtype=np.int)))

        # now split the data using sklearn stratified shuffle split
        shuffler = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.train_test_split,
            random_state=self.rand_seed).split(
            Goodchild_Combined_Serialised_Data,
            Goodchild_Combined_Serialised_Targets)
        indices = [(train_idx, validation_idx)
                   for train_idx, validation_idx in shuffler][0]
        # and now index the training data
        Goodchild_train_data = Goodchild_Combined_Serialised_Data[indices[0]]
        Goodchild_train_targets = Goodchild_Combined_Serialised_Targets[indices[0]]
        # and the testing data
        Goodchild_test_data = Goodchild_Combined_Serialised_Data[indices[1]]
        Goodchild_test_targets = Goodchild_Combined_Serialised_Targets[indices[1]]

        return Goodchild_train_data, Goodchild_train_targets, Goodchild_test_data, Goodchild_test_targets

    def get_params():
        return {"S_num": self.S_num,
                "UCL_lines_per_file": self.lines_per_UCL_file,
                "Goodchild_lines_per_file": self.lines_per_Goodchild_file,
                "Train_test_split": self.train_test_split,
                "random_seed": self.rand_seed}

    # private functions
    def __get_timestamp(self, filename):
        """
        Return timestamp extracted from filename.
        """
        idx = filename.find("_Channel_")
        if (idx == -1):
            return "Error!"

        return filename[0:idx].replace("_", " ")

    def __get_prn(self, filename):
        """
        Return satellite PRN code extracted from filename.
        """
        first = filename.find("_PRN") + len("_PRN")
        last = filename.find("_", first, len(filename))
        svid = int(filename[first:last])

        if svid < 10:
            return 'G0' + str(svid)
        else:
            return 'G' + str(svid)

    def __clear_sky_max(self, theta):
        """
        Return an "adjusted" cn0 based on the elevation angle of the ray.
        The adjustment is based on a fourth order polynomial found by
        least squares that relates the elevation angle to the peak CN0
        of the signal.
        """
        if (theta < 15):
            return np.polyval(cn0_elevation_coeffs, 15)

        if (theta > 80):
            # similarly, cap the upper limit.
            return np.polyval(cn0_elevation_coeffs, 80)

        return np.polyval(cn0_elevation_coeffs, theta)

    def __findLargestMultiple(self, target, divisor):
        """
        Returns the largest number less than or equal to target, but still
        divisible by divisor.
        """
        rem = target % divisor
        if(rem == 0):
            return target
        else:
            return target - rem

    def __load_serialise_normalise_file(
            self, data_loc, category, file, gt_path, s_num, num_rows=-1, normalise=False):
        # basic invariant check
        if (".csv" not in file):
            raise RuntimeError(
                "Invalid file. Can only open csv files of correlation data recorded using GNSS_SDR.")
            sys.exit(1)

        # read in the entire file as a dataframe
        try:
            if (num_rows == -1):
                # entirely sure that skipping bad lines is fine for the specific files
                # that I know are being read.
                temp = pd.read_csv(
                    path.join(
                        data_loc,
                        category,
                        file),
                    on_bad_lines='skip',
                    header=None)
            else:
                temp = pd.read_csv(path.join(data_loc, category, file), nrows=num_rows,
                                   on_bad_lines='skip', header=None)
        except IOError as e:
            print("Error opening csv file of correlation data. Error reported as: ", e)
            sys.exit(1)

        # drop na and normalise while still in panda form
        temp = temp.dropna()
        if (normalise):
            ts = self.__get_timestamp(file)
            svid = self.__get_prn(file)
            gts = pd.read_csv(gt_path)
            # search gt file for elevation angle
            elevation_angle = gts[(gts.time == ts) & (
                gts.svid == svid)].elevation_deg.values[0]

            # normalise using the predicted max CN0 at the given elevation
            temp = temp**2 / self.__clear_sky_max(elevation_angle)

        # now serialise the file data into groups of s_num
        max_row = self.__findLargestMultiple(temp.shape[0], s_num)
        # if there are two few rows in the file, then don't load the file's data
        # (may or may not be dubious, easier to ignore considering the quantity of data we're dealing with)
        if (max_row < s_num):
            return
        serialised_data = temp.iloc[0:max_row].to_numpy().reshape(
            max_row // s_num, s_num * temp.shape[1], 1)
#         serialised_data = TimeSeriesScalerMeanVariance(
#             mu=0.0, std=1.0).fit_transform(serialised_data)

        return serialised_data

    def __create_dataset(self, data_loc, category, files,
                         gt_path, s_num, num_rows=-1, normalise=False):
        arr = []
        # make sure to sort so that order is deterministic regardless of
        # underlying OS filesystem
        files.sort()
        for file in files:
            output = self.__load_serialise_normalise_file(
                data_loc, category, file, gt_path, s_num, num_rows, normalise)
            if (output is not None):
                arr.append(output)
            else:
                print("Removed blank entry")

        return np.concatenate(arr)

    def __load_and_test(self, data_loc, category, filenames,
                        gt_path, model, s_num, normalise=False):
        """
        Loads up the data file by file, serializes it, then generates a prediction
        for that particular ray (ray = instance of(timestamp + satellite))
        """
        # first, read in the ground truth file containing the elevations
        elevations = pd.read_csv(gt_path)
        predictions = []
        # go through each file. Each file corresponds to a single satellite and location
        # and timestamp (however, the length of the file is different from file
        # to file)
        for i in range(len(filenames)):
            # get the start timestamp and satellite svid of the file
            ts = self.__get_timestamp(filenames[i])
            svid = self.__get_prn(filenames[i])
            # now load up the data in the file
            serialised_data = self.__load_serialise_normalise_file(data_loc, category,
                                                              filenames[i], gt_path,
                                                              s_num)

            if (len(serialised_data) > 0):
                answers = model.predict(serialised_data)

            if (np.sum(answers) > len(answers) / 2):
                final_ans = 1

            else:
                final_ans = 0

            answer = {'timestamp': ts, 'svid': svid, 'prediction': final_ans}
            predictions.append(answer)

        return predictions
