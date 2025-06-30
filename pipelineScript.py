import os
import shutil
from preprocesser import *
from feature_extractor import *
from model_building import load_data, build_decision_tree, log_decision_tree, build_random_forest, results_saving_dt, results_saving_rf
from dataset_builder import compute_classes, build_dataset
from loggers import prompt_user_for_input
from pathlib import Path
from plotters import plot_preprocesspipeline_steps, plot_trap_functions
import time

"""Variables to run the experiments"""
to_preprocess = False  # To perform the preprocessing pipeline
to_extract_features = False  # To perform the first stage of the experiment pipeline
to_predict = False  # To perform the second stage of the experiment pipeline

"""Optional variables to run the experiments"""
to_plot = False  # Plot the results of the preprocessing pipeline - Optional - Beware of additional time and space consumption


"""Main script paths to save the files"""

input_files_parent_path = Path(f'input_data/')
experimentsOutput_path = Path(f'experimentsOutput/')
experimentsInput_path = Path(f'experimentsInput/')
model = 'dt'  # 'dt' for Decision Tree, 'rf' for Random Forest
defuzzification_function = 'mom' # 'mom', 'som', 'lom', 'centroid', 'bisector'


#Defatult values for the experiments - Comment the next line to run the script without user input

"""year = '2019'
region_path = 'North'
dataset = 'ds3'"""

year,region_path,dataset = prompt_user_for_input()



pipeline_path = Path(f'{experimentsOutput_path}/{region_path}/{year}')
input_files_path = Path(f'{input_files_parent_path}/{region_path}/')

"""File paths to save the results of the prerocess pipeline"""

experiments_output_parent_path = Path(f'{experimentsOutput_path}/{region_path}/{year}/')

daily_averages_u_path = Path(f'{experiments_output_parent_path}/daily_averages/u')
daily_averages_v_path = Path(f'{experiments_output_parent_path}/daily_averages/v')
rotated_data_path = Path(f'{experiments_output_parent_path}/rotated_data')
zoomed_data_path = Path(f'{experiments_output_parent_path}/zoomed_data')
plot_output_path = Path(f'{experiments_output_parent_path}/plots')
model_building_results_path = Path(f'{experiments_output_parent_path}/{dataset}')



# Check if pipeline folder exists
if os.path.exists(pipeline_path):
    print('Pipeline folder with the name', pipeline_path)
    print('Do you want to overwrite the folder? Type Y for yes or N for no')

    while True:
        answer = input()
        if answer == 'Y':
            print('Deleting the folder')
            shutil.rmtree(pipeline_path)
            break
        elif answer == 'N':
            print('Not deleting the folder')
            break
        else:
            print('Invalid input. Type Y for yes or N for no')
else:
    print('Pipeline folder does not exist')
    print('Creating the folder')
    os.makedirs(pipeline_path)


if region_path == 'North':
    x = 251
    y = 501
else:
    x = 351
    y = 426


start_time = time.time()

if to_preprocess:

    """shutil.rmtree(daily_averages_u_path, ignore_errors=True)
    shutil.rmtree(daily_averages_v_path, ignore_errors=True)
    shutil.rmtree(rotated_data_path, ignore_errors=True)
    shutil.rmtree(zoomed_data_path, ignore_errors=True)


    os.makedirs(daily_averages_u_path)
    os.makedirs(daily_averages_v_path)
    os.makedirs(rotated_data_path)
    os.makedirs(zoomed_data_path)"""

    print("Opening grib file")
    u_values, v_values, lats, lons, length = read_data(input_files_path, year, region_path)

    print("Preprocessing grib file")
    #Get daily u and v average values
    u_values, v_values = daily_averages(u_values, v_values, length, daily_averages_u_path, daily_averages_v_path)

    #Rotate u and v average values
    parallel_component = rotate_vectors(u_values, v_values, rotated_data_path)

    #Zoom on the parallel component
    parallel_component = zoom_parallel_component(parallel_component,input_files_path,zoomed_data_path)

    print("Finished preprocessing grib file")


if to_plot:
    shutil.rmtree(plot_output_path, ignore_errors=True)
    os.makedirs(plot_output_path)

    print("Plotting the preprocessing pipeline steps")
    plot_preprocesspipeline_steps(year,region_path, lons, lats,input_files_path,daily_averages_u_path, daily_averages_v_path, rotated_data_path, zoomed_data_path, plot_output_path)

if to_extract_features:

    print("Extracting features from the zoomed data")

    # check if extracted_features.csv exists, if it does, skip

    if not os.path.exists(experiments_output_parent_path / 'extracted_features.csv'):
        # Extract features from the zoomed data if the file is non-existent
        extract_features(zoomed_data_path, input_files_path, x, y, region_path, year, experiments_output_parent_path)

    # Build the dataset
    trap_functions, cores, centroids = compute_classes(year, region_path, experimentsInput_path, defuzzification_function)
    build_dataset(experiments_output_parent_path, experimentsInput_path, year, region_path, experimentsInput_path, defuzzification_function)

    # Plot the trapezoidal functions
    plot_trap_functions(trap_functions, cores, centroids, year, region_path, defuzzification_function, plot_output_path)


if to_predict:

    print("Building the models")
    if not os.path.exists(model_building_results_path):
        os.makedirs(model_building_results_path)

    rule_list = []
    stats_list = []

    features, labels = load_data(experimentsInput_path, year, region_path, dataset)

    if model == 'dt':
        # Test for depths 3 to 10
        for depth in range(3,11):
            model, model_stats, class_names = build_decision_tree(features, labels, depth)
            rules = log_decision_tree(model, class_names, depth, region_path, year, defuzzification_function, model_building_results_path)
            for r in rules:
                rule_list.append(r)
            stats_list.append(model_stats)
        # Save the results
        results_saving_dt(rule_list, stats_list, year, region_path, defuzzification_function, model_building_results_path)
    else:
        # Build the random forest model
        model,stats_list = build_random_forest(features, labels)
        results_saving_rf(stats_list, year, region_path, model_building_results_path, model, defuzzification_function, dataset)


print("Pipeline concluded. The results are saved in the folder",experiments_output_parent_path)

end_time = time.time()

print("Total time taken to run the pipeline: {:.2f} seconds".format(end_time - start_time))


