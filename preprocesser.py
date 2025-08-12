from pathlib import Path

import numpy as np
import xarray
from scipy.io import loadmat
from scipy.ndimage import zoom



def read_data(input_files_path: Path,year: str,region_sst: str) -> tuple[np.array, np.array, np.array, np.array, int]:

    """Function for reading data from grib files
    @:param input_files_path: Path to the directory containing the grib files
    @:param year: Year for which the data is to be read
    @:param region_sst: Region for which the SST data is to be read
    @:return: tuple of numpy arrays containing u and v wind components, latitudes, longitudes and the length of the data"""

    path_wind = input_files_path / f'{year}/{year}_FULL_{region_sst}.grib'
    path_sst = input_files_path / f'sst-{region_sst.lower()}.grib'

    data = xarray.open_mfdataset(path_wind, engine='cfgrib', parallel=True)
    data_sst = xarray.open_mfdataset(path_sst, engine='cfgrib', parallel=True)

    uwind = data['u10']
    vwind = data['v10']
    file_length = len(data['u10'].values)

    lats, lons = data['latitude'].values, data['longitude'].values
    lons, lats = np.meshgrid(lons, lats)

    values_u = uwind[:].values
    values_v = vwind[:].values

    sst = data_sst['sst'].values
    values_u[:] = np.where(np.isnan(sst[0]), np.nan, values_u)
    values_v[:] = np.where(np.isnan(sst[0]), np.nan, values_v)

    return values_u,values_v,lats,lons,file_length




def daily_averages(u_data: np.ndarray, v_data: np.ndarray, length: int,daily_averages_u_path: Path, daily_averages_v_path: Path)-> tuple[list, list]:

    """Computes daily averages of u and v wind components
    @:param u_data: numpy array of u wind component data
    @:param v_data: numpy array of v wind component data
    @:param length: length of the data
    @:param daily_averages_u_path: path to save daily averages of u wind component
    @:param daily_averages_v_path: path to save daily averages of v wind component
    @:return: tuple of lists containing daily averages of u and v wind components"""

    u_avg_list = []
    v_avg_list = []
    for i in range(0, length, 24):
        u = u_data[i:i + 24]
        u = np.nanmean(u, axis=0)
        v = v_data[i:i + 24]
        v = np.nanmean(v, axis=0)
        u_avg_list.append(u)
        v_avg_list.append(v)
        day_num = int(i/24)
        np.savetxt(daily_averages_u_path / f'day_{str(day_num)}.csv', u, delimiter=',')
        np.savetxt(daily_averages_v_path / f'day_{str(day_num)}.csv', v, delimiter=',')

    return u_avg_list, v_avg_list



def rotate_vectors(u_data: list, v_data: list, rotated_data_path: Path, coastline_angle=-55) -> list:

    """Rotates the u and v wind components based on the coastline angle
    @:param u_data: list of numpy arrays containing u wind component data
    @:param v_data: list of numpy arrays containing v wind component data
    @:param rotated_data_path: path to save the rotated data
    @:param coastline_angle: angle of the coastline in degrees (default is -55)
    @:return: list of rotated u wind component data"""

    rotate_matrices = []
    for u, v in zip(u_data, v_data):
        rotate_matrices.append(u * np.cos(np.deg2rad(coastline_angle)) - v * np.sin(np.deg2rad(coastline_angle)))
        # save the rotated data to a file
        np.savetxt(rotated_data_path / f'day_{str(len(rotate_matrices)-1)}.csv', rotate_matrices[-1], delimiter=',')

    return rotate_matrices



def zoom_parallel_component(average_parallel_component: list,input_files_path: Path,zoomed_data_path: Path)-> None:

    """Zooms the parallel component data to match the shape of the SST data
    @:param average_parallel_component: list of numpy arrays containing the average parallel component data
    @:param input_files_path: path to the input files
    @:param zoomed_data_path: path to save the zoomed data
    @:return: None"""

    sst_data = loadmat(input_files_path / f'window_5_n_0.mat')['imagem']
    #sst_data = np.flipud(sst_data)
    sst_data_shape = sst_data.shape

    for i in range(len(average_parallel_component)):
        wind = average_parallel_component[i]
        wind = wind[4:,:]
        zoom_factor_x = sst_data_shape[1]/wind.shape[1]
        zoom_factor_y = sst_data_shape[0]/wind.shape[0]
        wind[np.isnan(wind)] = 0
        zoom_wind = zoom(wind, (zoom_factor_y, zoom_factor_x))
        zoom_wind[np.isnan(sst_data)] = np.nan
        #flat_zoom_nan = zoom_wind.flatten()
        flat_zoom_nan = np.abs(zoom_wind)
        np.savetxt(zoomed_data_path / f'average_parallel_component_{i}_nan.csv', flat_zoom_nan, delimiter=',')

        """flat_zoomed = flat_zoom_nan[~np.isnan(flat_zoom_nan)]
        flat_zoomed = flat_zoomed.reshape((flat_zoomed.shape[0], 1))
        np.savetxt(zoomed_data_path / f'average_parallel_component_{i}.csv', flat_zoomed, delimiter=',')"""


