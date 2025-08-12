import numpy as np
import skfuzzy as fuzz
from pathlib import Path
import pandas as pd

"""Dictionary with the instants of the USP for each region and year"""
usp_instants_dict = {
    'north': {
        '2004': [1, 14, 22, 25],
        '2005': [1, 16, 29],
        '2006': [1, 12, 16, 20],
        '2007': [1, 12, 23],
        '2008': [1, 13, 20, 39],
        '2009': [1, 18, 21],
        '2010': [1, 5, 22, 27],
        '2011': [1, 7, 18, 21],
        '2012': [1, 12, 18, 26],
        '2013': [1, 9, 21, 33],
        '2014': [1, 12, 25],
        '2015': [1, 9, 22, 26],
        '2016': [1, 11, 18, 26],
        '2017': [1, 7, 20],
        '2018': [1, 8, 20, 24],
        '2019': [1, 12, 21, 40]
    },
    'south': {
        '2004': [1, 20, 23, 34],
        '2005': [1, 16, 23, 37],
        '2006': [1, 20, 26],
        '2007': [1, 19, 24, 33],
        '2008': [1, 19, 23, 33],
        '2009': [1, 19, 24, 28],
        '2010': [1, 22, 26],
        '2011': [1, 18, 22, 25, 38],
        '2012': [1, 18, 23, 39],
        '2013': [1, 19, 24, 30],
        '2014': [1, 20, 26, 34],
        '2015': [1, 13, 16, 23, 31, 37],
        '2016': [1, 5, 20, 26],
        '2017': [1, 16, 22, 29],
        '2018': [1, 20, 36],
        '2019': [1, 15, 22, 25, 32]
    }
}

header_0 = 'Z1_Z2_Difference,Z4_Z5_Difference,Z7_Z9_Difference,Z10_Z12_Difference,'
header_1 = 'Min_WSA_Z1,Avg_WSA_Z1,Max_WSA_Z1,'
header_2 = 'Min_WSA_Z2,Avg_WSA_Z2,Max_WSA_Z2,'
header_3 = 'Min_WSA_Z4,Avg_WSA_Z4,Max_WSA_Z4,'
header_4 = 'Min_WSA_Z5,Avg_WSA_Z5,Max_WSA_Z5,'
header_5 = 'Min_WSA_Z7,Avg_WSA_Z7,Max_WSA_Z7,'
header_6 = 'Min_WSA_Z8,Avg_WSA_Z8,Max_WSA_Z8,'
header_7 = 'Min_WSA_Z9,Avg_WSA_Z9,Max_WSA_Z9,'
header_8 = 'Min_WSA_Z10,Avg_WSA_Z10,Max_WSA_Z10,'
header_9 = 'Min_WSA_Z11,Avg_WSA_Z11,Max_WSA_Z11,'
header_10 = 'Min_WSA_Z12,Avg_WSA_Z12,Max_WSA_Z12,'
header_11 = 'Min_Temp_Z1,Avg_Temp_Z1,Max_Temp_Z1,'
header_12 = 'Min_Temp_Z2,Avg_Temp_Z2,Max_Temp_Z2,'
header_13 = 'Min_Temp_Z4,Avg_Temp_Z4,Max_Temp_Z4,'
header_14 = 'Min_Temp_Z5,Avg_Temp_Z5,Max_Temp_Z5,'
header_15 = 'Min_Temp_Z7,Avg_Temp_Z7,Max_Temp_Z7,'
header_16 = 'Min_Temp_Z8,Avg_Temp_Z8,Max_Temp_Z8,'
header_17 = 'Min_Temp_Z9,Avg_Temp_Z9,Max_Temp_Z9,'
header_18 = 'Min_Temp_Z10,Avg_Temp_Z10,Max_Temp_Z10,'
header_19 = 'Min_Temp_Z11,Avg_Temp_Z11,Max_Temp_Z11,'
header_20 = 'Min_Temp_Z12,Avg_Temp_Z12,Max_Temp_Z12,Class'

"""Header for the dataset"""
headers = header_1+header_2+header_3+header_4+header_5+header_6+header_7+header_8+header_9+header_10+header_0+header_11+header_12+header_13+header_14+header_15+header_16+header_17+header_18+header_19+header_20





def find_first_week_day(week_number: int, days_array: [], num_days_week=8):
    """Finds the first day of the week given the week number and the array of days
    @:param week_number: number of the week (default is 8)
    @:param days_array: array with the days of the year by order
    @:param num_days_week: number of days in a week"""

    if week_number == 46:
        return days_array[num_days_week * week_number - 3]
    return days_array[num_days_week * week_number - 8]





def find_last_week_day(week_number: int, days_array: [], num_days_week=8):

    """Finds the last day of the week given the week number and the array of days
    @:param week_number: number of the week
    @:param days_array: array with the days of the year by order
    @:param num_days_week: number of days in a week"""

    if week_number == 46:
        return days_array[num_days_week * week_number - 3]
    return days_array[num_days_week * week_number - 1]




def find_trap_support(instants_array: [], days_array: [], initial_instant: int, final_instant: int):
    """Finds the support of the trapezoid function
    @:param instants_array: array with the instants indicating the first and last week of the respective instant
    @:param days_array: array with the days of the year by order
    @:param initial_instant: initial instant of the time-range
    @:param final_instant: final instant of the time-range"""


    first_week, last_week = instants_array[initial_instant - 1][0], instants_array[final_instant - 1][1]
    first_day, last_day = find_first_week_day(first_week, days_array), find_last_week_day(last_week, days_array)
    return first_day, last_day


def find_core(previous_support: [], current_support: [], next_support: []):

    """Finds the core of the trapezoid function
    @:param previous_support: tuple with the support of the previous instant
    @:param current_support: tuple with the support of the current instant
    @:param next_support: tuple with the support of the next instant"""

    if previous_support == current_support:
        return 0, next_support[0]
    if current_support == next_support:
        return previous_support[1], 366
    return previous_support[1], next_support[0]


days = np.arange(0, 366, 1)
weeks = np.arange(1, 46, 1)
instants = np.column_stack((weeks, weeks + 4))


def define_trapezoid(x: [], a: int, b: int, c: int, d: int) -> np.array:
    """Defines the trapezoid funciton
    @:param x: array of values
    @;param a: initial support
    @:param b: initial core
    @:param c: final core
    @:param d: final support"""

    return fuzz.trapmf(x, [a, b, c, d])



def classify(day: int, centroids: np.array(int)) -> int:

    """Classifies a day based on the centroids of the trapezoidal functions
    @:param day: day of the year (0-365)
    @:param centroids: array of centroids of the trapezoidal functions
    @:return: assigned class"""

    assigned_class = np.argmin(np.abs(day - centroids)) + 1
    return assigned_class



def compute_trap_functions(usp_instants: list) -> list:

    """Computes the trapezoidal functions based on the USP instants
    @:param usp_instants: list of USP instants for a specific region and year
    @:param days: array of days in the year
    @:param instants: array of instants indicating the first and last week of the respective instant
    @:return: list of trapezoidal functions for the given USP instants"""

    trap_functions = []
    supports = []
    cores = []
    #compute set of supports
    for i in range(len(usp_instants)):
        if i + 1 == len(usp_instants):
            supports.append(find_trap_support(instants, days, usp_instants[i], 42))
        else:
            supports.append(find_trap_support(instants, days, usp_instants[i], usp_instants[i + 1] - 1))

    #compute set of cores and compute trapezoidal functions

    for i in range(len(supports)):
        if i == 0:
            core_start, core_end = find_core(supports[i], supports[i], supports[i + 1])
        elif i + 1 == len(supports):
            core_start, core_end = find_core(supports[i - 1], supports[i], supports[i])
        else:
            core_start, core_end = find_core(supports[i - 1], supports[i], supports[i + 1])


        if core_start > core_end:
            core_start = core_end

        cores.append((core_start, core_end))

        if i + 1 == len(supports):
            trap_functions.append(define_trapezoid(days, supports[i][0], core_start, core_end, supports[i][1] + 1))
        else:
            trap_functions.append(define_trapezoid(days, supports[i][0], core_start, core_end, supports[i][1]))

    return trap_functions, cores



def compute_centroids(trap_functions: list, days: [], mode) -> list:

    """Computes the centroids of the trapezoidal functions
    @:param trap_functions: list of trapezoidal functions
    @:param days: array of days in the year
    @:param mode: mode of the defuzzification function (e.g., 'centroid', 'bisector', 'mom')"""

    centroids = []
    for i in range(len(trap_functions)):
        centroids.append(fuzz.defuzz(days, trap_functions[i], mode))
    return centroids



def compute_classes(year: str, region: str, output_path: Path, mode: str) -> tuple:
    """Computes the classes for each day of the year based on the USP instants
    @:param year: Year of the dataset
    @:param region: Region of the dataset
    @:param output_path: Path to save the computed classes
    @:param mode: Mode of the defuzzification function (e.g., 'centroid', 'bisector', 'mom')"""

    try:
        usp_instants = usp_instants_dict[region.lower()][year]
        trap_functions, cores = compute_trap_functions(usp_instants)
        centroids = np.array(compute_centroids(trap_functions, days, mode))
        usp = np.vstack([classify(i, centroids) for i in range(0, 365)])

        np.savetxt(output_path / f'{year}_{region}_{mode}.csv', usp, delimiter=',', fmt='%1.0f')
        return trap_functions, cores, centroids
    except AssertionError as e:
        print('Error in:', year, region, mode)
        print(e)
        print('-------------------')



def build_dataset(extracted_features_path: Path, classes_path: Path, year: str, region: str, output_path: Path, mode: str) -> None:
    """Function to merge the extracted features and classes into a final dataset
    @:param extracted_features_path: Path to the folder containing the extracted features
    @:param classes_path: Path to the folder containing the classes
    @:param year: Year of the dataset
    @:param region: Region of the dataset
    @:param output_path: Path to save the final dataset
    @:param mode: Mode of the dataset (e.g., 'mom', 'defuzzification_function')"""

    df_features = pd.read_csv(extracted_features_path / f'extracted_features.csv', header=None)
    df_classes = pd.read_csv(classes_path / f'{year}_{region}_{mode}.csv', header=None)
    df_features.fillna(df_features.mean(), inplace=True)




    df_features = df_features.round(2)

    final_dataset = pd.concat([df_features, df_classes], axis=1).to_numpy()
    np.savetxt(output_path / f'{year}_{region}_{mode}.csv', final_dataset, header=headers,delimiter=',')

    #add a dummy row to the dataset at the end
    df_features = df_features.append(pd.Series(), ignore_index=True)
    df_features = df_features.append(pd.Series(), ignore_index=True)

    #compute minimum, average and maximum values of the dataset per each column
    dataset_min = df_features.min()
    dataset_avg = df_features.mean().round(2)
    dataset_std = df_features.std().round(2)
    dataset_max = df_features.max()

    #merge each coresponding value of mean and standard deviation into a single df where each cell has the format mean +- stddev
    dataset_avg = dataset_avg.astype(str) + ' ± ' + dataset_std.astype(str)

    dataset_mode = df_features.mode().iloc[0]  # Get the mode for each column, taking the first mode in case of multiple modes
    dataset_median = df_features.median()  # Get the median for each column

    #add rows to the original dataset so that it contains the min, avg, max, mode and median values
    df_features = df_features.append(dataset_min, ignore_index=True)
    df_features = df_features.append(dataset_avg, ignore_index=True)
    df_features = df_features.append(dataset_max, ignore_index=True)
    df_features = df_features.append(dataset_mode, ignore_index=True)
    df_features = df_features.append(dataset_median, ignore_index=True)

    #add a dummy column to the dataset as the first column with the last 5 rows being the min, avg, max, mode and median names
    df_features.insert(0, 'Statistics', [''] * (len(df_features) - 5) + ['Min', 'Avg ± Std', 'Max', 'Mode', 'Median'])

    #save the dataset to a new csv file
    df_features.to_excel(extracted_features_path / f'extracted_features_summary.xlsx', index=False,float_format='%.2f', header=headers)
