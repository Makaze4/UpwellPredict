from pathlib import Path
import numpy as np
import math
from scipy.io import loadmat


def extract_features(parallel_component_path: Path,sst_input_data_path: Path,x: int,y: int,region: str,year: str,extracted_features_output_path: Path,c=1.8375e-3)-> np.ndarray:

    """Function to extract features from wind stress anomaly and sea surface temperature data.
    @:param parallel_component_path: Path to the directory containing the parallel component data.
    @:param sst_input_data_path: Path to the directory containing sea surface temperature data.
    @:param x: horizontal shape of a sst grid.
    @:param y: vertical shape of a sst grid.
    @:param region: Region for which the data is being processed (e.g., 'North', 'South').
    @:param year: Year for which the data is being processed.
    @:param extracted_features_output_path: Path to save the extracted features.
    @:param c: Constant to be used in the wind stress calculation (default is 1.8375e-3).
    @:return: A numpy array containing the extracted features for each day."""

    zone_difference = []
    zones_stats_df = []

    for k in range(0,366):
        if k == 365 and year not in ['2004', '2008', '2012', '2016']:
            break
        wind_data = np.loadtxt(parallel_component_path / f'average_parallel_component_{k}_nan.csv', delimiter=',')
        #wind_data = np.loadtxt(parallel_component_path / f'day_{k}.csv', delimiter=',')
        wind_data = wind_data.reshape(x,y)

        wind_stress_data = wind_data**2 # * c - Uncomment to add the constant to wind stress
        wind_stress_anomaly_data = wind_stress_data - np.nanmean(wind_stress_data)

        sst_data = np.flipud(loadmat(sst_input_data_path / f'{region.lower()}_morocco_{year}_2km_{int(k/8)}.mat')['imagem'])
        index_slices = math.ceil(wind_data.shape[0]/4), math.ceil(wind_data.shape[1]/3)

        zone_stats = []
        zone_stats_temp = []
        counter = 1

        for i in range(0,4):
            for j in range(0,3):
                x_start = wind_data.shape[0] - (i + 1) * index_slices[0]
                x_end = wind_data.shape[0] - i * index_slices[0]
                #y_start = j * y_size
                #y_end = min((j + 1) * y_size, sst_file.shape[1])  # stay in bounds

                # Clip negative indices in case we're at the top row
                x_start = max(0, x_start)

                #sst_file.shape[0] - (i + 1) * x_size
                #sst_file.shape[0] - i * x_size
                #data_to_select_sst = sst_data[wind_data.shape[0] - (i+1) * index_slices[0]: wind_data.shape[0] - i * index_slices[0], j*index_slices[1] : (j+1)*index_slices[1]]
                data_to_select = wind_stress_anomaly_data[x_start:x_end, j*index_slices[1] : (j+1)*index_slices[1]]
                data_to_select_sst = sst_data[x_start:x_end, j*index_slices[1] : (j+1)*index_slices[1]]
                #data_to_select = wind_stress_anomaly_data[i*index_slices[0]: (i+1)*index_slices[0], j*index_slices[1] : (j+1)*index_slices[1]]

                number_not_nans, number_nans = np.count_nonzero(~np.isnan(data_to_select)), np.count_nonzero(np.isnan(data_to_select))

                if ((number_not_nans * 100) / data_to_select.size) < 1:
                    counter += 1
                    continue


                #append WSA min,avg,max values
                zone_stats.append(np.nanmin(data_to_select))
                zone_stats.append(np.nanmean(data_to_select))
                zone_stats.append(np.nanmax(data_to_select))

                #append Temperature min,avg,max values
                zone_stats_temp.append(np.nanmin(data_to_select_sst))
                zone_stats_temp.append(np.nanmean(data_to_select_sst))
                zone_stats_temp.append(np.nanmax(data_to_select_sst))

                zone_difference.append(np.nanmean(data_to_select_sst))

                counter += 1

        # Zone 1 - Zone 2
        zone_stats.append(zone_difference[0] - zone_difference[1])
        # Zone 4 - Zone 5
        zone_stats.append(zone_difference[2] - zone_difference[3])
        # Zone 7 - Zone 9
        zone_stats.append(zone_difference[4] - zone_difference[6])
        # Zone 10 - Zone 12
        zone_stats.append(zone_difference[7] - zone_difference[9])

        zone_stats.extend(zone_stats_temp)
        zone_stats = np.array(zone_stats)
        zones_stats_df.append(zone_stats)
        zone_difference = []
    zones_stats_df = np.vstack(zones_stats_df)

    np.savetxt(extracted_features_output_path / f'extracted_features.csv',zones_stats_df,delimiter=',')

    return zones_stats_df
