import matplotlib.pyplot as plt
import pandas as pd
import xarray
from metpy.units import units
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.io import loadmat
from scipy.ndimage import zoom
from sklearn.tree import plot_tree
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from mpl_toolkits.basemap import Basemap
import metpy.calc as mpcalc
import calendar


def generate_month_days() -> list:
    """Generates a list of strings representing the days of each month in the format 'day-month'.
    Each month is represented by its abbreviated name (e.g., 'Jan', 'Feb', etc.).
    Returns:
        list: A list of strings in the format 'day-month', where 'day' is the day of the month and 'month' is the abbreviated month name.
    """

    # Initialize an empty list to store the day strings
    day_strings = []

    # Iterate over the months from April (4) to October (10)
    for month in range(1, 13):
        # Get the number of days in the current month
        num_days = calendar.monthrange(2024, month)[1]  # Assuming year 2024

        # Create day strings for each day in the month
        for day in range(1, num_days + 1):
            day_strings.append(f"{day}-{calendar.month_abbr[month]}")

    return day_strings


def get_region_angles(angles,region_path)->list:
    """Adjusts the angles based on the region path.
    @:param angles: The angles to be adjusted.
    @:param region_path: The region path to determine the adjustment.
    @:return: The adjusted angles."""

    if region_path == 'North':
        angles[(angles >= 30) & (angles <= 30.25)] = np.deg2rad(140)
        angles[(angles > 30.25) & (angles <= 31.25)] = np.deg2rad(180)
        angles[(angles > 31.25) & (angles <= 32.25)] = np.deg2rad(209)
        angles[(angles > 32.25) & (angles <= 33.25)] = np.deg2rad(221)
        angles[(angles > 33.25) & (angles <= 34)] = np.deg2rad(240)
        angles[(angles > 34) & (angles <= 36)] = np.deg2rad(202)

        """angles[(angles >= 30) & (angles <= 30.25)] = np.deg2rad(320)
        angles[(angles > 30.25) & (angles <= 31.25)] = np.deg2rad(0)
        angles[(angles > 31.25) & (angles <= 32.25)] = np.deg2rad(389)
        angles[(angles > 32.25) & (angles <= 33.25)] = np.deg2rad(401)
        angles[(angles > 33.25) & (angles <= 34)] = np.deg2rad(420)
        angles[(angles > 34) & (angles <= 36)] = np.deg2rad(382)"""


    else:
        angles[(angles >= 20) & (angles <= 20.5)] = np.deg2rad(324)
        angles[(angles > 20.5) & (angles <= 22.25)] = np.deg2rad(370)
        angles[(angles > 22.25) & (angles <= 24.5)] = np.deg2rad(390)
        angles[(angles > 24.5) & (angles < 26.25)] = np.deg2rad(380)
        angles[(angles >= 26.25) & (angles <= 26.75)] = np.deg2rad(420)
        angles[(angles > 26.75) & (angles <= 28)] = np.deg2rad(384)
    return angles

def get_region_angles_2(angles,region_path)->list:
    """Adjusts the angles based on the region path.
    @:param angles: The angles to be adjusted.
    @:param region_path: The region path to determine the adjustment.
    @:return: The adjusted angles."""

    if region_path == 'North':
        """angles[(angles >= 30) & (angles <= 30.25)] = np.deg2rad(140)
        angles[(angles > 30.25) & (angles <= 31.25)] = np.deg2rad(180)
        angles[(angles > 31.25) & (angles <= 32.25)] = np.deg2rad(209)
        angles[(angles > 32.25) & (angles <= 33.25)] = np.deg2rad(221)
        angles[(angles > 33.25) & (angles <= 34)] = np.deg2rad(240)
        angles[(angles > 34) & (angles <= 36)] = np.deg2rad(202)"""

        angles[(angles >= 30) & (angles <= 30.25)] = np.deg2rad(320)
        angles[(angles > 30.25) & (angles <= 31.25)] = np.deg2rad(0)
        angles[(angles > 31.25) & (angles <= 32.25)] = np.deg2rad(389)
        angles[(angles > 32.25) & (angles <= 33.25)] = np.deg2rad(401)
        angles[(angles > 33.25) & (angles <= 34)] = np.deg2rad(420)
        angles[(angles > 34) & (angles <= 36)] = np.deg2rad(382)


    else:
        angles[(angles >= 20) & (angles <= 20.5)] = np.deg2rad(144)
        angles[(angles > 20.5) & (angles <= 22.25)] = np.deg2rad(190)
        angles[(angles > 22.25) & (angles <= 24.5)] = np.deg2rad(210)
        angles[(angles > 24.5) & (angles < 26.25)] = np.deg2rad(200)
        angles[(angles >= 26.25) & (angles <= 26.75)] = np.deg2rad(240)
        angles[(angles > 26.75) & (angles <= 28)] = np.deg2rad(204)
    return angles

def plot_preprocesspipeline_steps(year: str,region_path: str,lons: np.array, lats: np.array, input_files_path:Path, u_path: Path, v_path: Path, rotated_path: Path, zoomed_path: Path, save_plot_path: Path)-> None:

    sst_data = xarray.open_mfdataset(input_files_path / f'sst-{region_path}.grib', engine='cfgrib', parallel=True)

    sst_data_values = sst_data['sst'].values[0][4:, :]

    lats = lats[4:, :]
    lons = lons[4:, :]

    #find the overall maximum and minimum of lats and lons
    max_lat = np.nanmax(lats)
    min_lat = np.nanmin(lats)
    max_lon = np.nanmax(lons)
    min_lon = np.nanmin(lons)





    sst = loadmat(input_files_path / f'window_5_n_0.mat')['imagem']
    sst_data = np.flipud(sst)
    sst_shape = sst_data.shape

    x_new = np.linspace(min_lon, max_lon, sst_shape[1])
    y_new = np.linspace(min_lat, max_lat, sst_shape[0])
    xv_new, yv_new = np.meshgrid(x_new, y_new)

    for i in range(8):
        if i == 365 and year not in ['2004', '2008', '2012', '2016']:
            break

        # --- Loading Files ---
        u = np.loadtxt(u_path / f'day_{i}.csv', delimiter=',')[4:, :]
        v = np.loadtxt(v_path / f'day_{i}.csv', delimiter=',')[4:, :]
        wind = np.loadtxt(rotated_path / f'day_{i}.csv', delimiter=',')[4:, :]
        #zoomed_wind = np.loadtxt(zoomed_path / f'average_parallel_component_{i}_nan.csv', delimiter=',').reshape(sst_shape)


        # --- Plotting ---
        fig, axes = plt.subplots(1, 4, figsize=(32, 4), sharex=True, sharey=True)

        cax0 = inset_axes(axes[0],
                          width="5%",        # Width of colorbar
                          height="100%",     # Height relative to the axes
                          loc='right',
                          borderpad=0)


        cax1 = inset_axes(axes[1],
                          width="5%",        # Width of colorbar
                          height="100%",     # Height relative to the axes
                          loc='right',
                          borderpad=0)
        cax2 = inset_axes(axes[2],
                          width="5%",        # Width of colorbar
                          height="100%",     # Height relative to the axes
                          loc='right',
                          borderpad=0)
        cax3 = inset_axes(axes[3],
                          width="5%",        # Width of colorbar
                          height="100%",     # Height relative to the axes
                          loc='right',
                          borderpad=0)


        # Original plot
        bmap0 = Basemap(projection='cyl',
                        llcrnrlat=min_lat, urcrnrlat=max_lat,
                        llcrnrlon=min_lon, urcrnrlon=max_lon,
                        ax=axes[0])
        bmap0.drawcoastlines()

        magnitude = np.sqrt(u ** 2 + v ** 2)

        # plot the magnitude for a region given by the latitudes and longitudes
        cf0 = bmap0.contourf(lons, lats, magnitude, latlon=True, cmap=plt.cm.jet)

        axes[0].set_xticks(np.arange(lons.min(), lons.max(), 2))
        axes[0].set_yticks(np.arange(lats.min(), lats.max(), 2))

        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')

        axes[0].set_title('Original Wind plot with wind vectors')

        # plot the wind vectors
        bmap0.quiver(lons[::2, ::2], lats[::2, ::2], u[::2, ::2], v[::2, ::2], scale=150, color='k')
        plt.colorbar(cf0, cax=cax0, orientation='vertical',label='Wind Speed (m/s)')


        #Rotated plot

        angles = np.copy(lats)

        if region_path == 'North':
            angles = get_region_angles(angles,region_path)
        else:
            angles = get_region_angles_2(angles,region_path)

        u_value,v_value = mpcalc.wind_components(wind * units('m/s'), angles * units('rad'))

        wind[np.isnan(sst_data_values)] = np.nan



        bmap1 = Basemap(projection='cyl',
                        llcrnrlat=min_lat, urcrnrlat=max_lat,
                        llcrnrlon=min_lon, urcrnrlon=max_lon,
                        ax=axes[1])

        cf1 = bmap1.contourf(lons, lats, np.abs(wind), cmap='jet', latlon=True)

        bmap1.drawcoastlines()

        axes[1].set_title('Alongshore Wind Component with wind vectors')
        plt.colorbar(cf1, cax=cax1, orientation='vertical',label='Wind Speed (m/s)')

        axes[1].set_xticks(np.arange(min_lon, max_lon, 2))
        axes[1].set_yticks(np.arange(min_lat, max_lat, 2))

        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')

        # plot the wind vectors
        bmap1.quiver(lons[::2, ::2], lats[::2, ::2], u_value[::2, ::2], v_value[::2, ::2], scale=150, color='k')


        # Zoomed plot


        angles = np.copy(yv_new)

        if region_path == 'North':
            angles = get_region_angles_2(angles,region_path)
        else:
            angles = get_region_angles(angles,region_path)



        zoom_factor_x = sst_shape[1]/wind.shape[1]
        zoom_factor_y = sst_shape[0]/wind.shape[0]
        wind[np.isnan(wind)] = 0
        zoom_wind = zoom(wind, (zoom_factor_y, zoom_factor_x))
        zoom_wind[np.isnan(sst_data)] = np.nan
        #flat_zoom_nan = zoom_wind.flatten()
        flat_zoom_nan = np.abs(zoom_wind).reshape(sst_shape)


        u_value,v_value = mpcalc.wind_components(np.flipud(flat_zoom_nan) * units('m/s'), angles * units('rad'))


        bmap2 = Basemap(projection='cyl',
                        llcrnrlat=yv_new.min(), urcrnrlat=yv_new.max(),
                        llcrnrlon=xv_new.min(), urcrnrlon=xv_new.max(),
                        ax=axes[2])

        cf2 = bmap2.contourf(xv_new, yv_new, np.flipud(flat_zoom_nan), cmap='jet', latlon=True)
        axes[2].set_title('Zoomed Parallel Component with wind vectors')
        plt.colorbar(cf2, cax=cax2, orientation='vertical',label='Wind Speed (m/s)')

        axes[2].set_xticks(np.arange(lons.min(), lons.max(), 2))
        axes[2].set_yticks(np.arange(lats.min(), lats.max(), 2))

        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')

        bmap2.quiver(xv_new[::20, ::22], yv_new[::20, ::22], u_value[::20, ::22], v_value[::20, ::22], scale=150, color='k')


        # WSA plot
        wind_sq = flat_zoom_nan**2

        wind_sq_mean = np.nanmean(wind_sq)

        wsa = wind_sq-wind_sq_mean

        bmap3 = Basemap(projection='cyl',
                        llcrnrlat=min_lat, urcrnrlat=max_lat,
                        llcrnrlon=min_lon, urcrnrlon=max_lon,
                        ax=axes[3])

        #bmap3.drawcoastlines()

        cf3 = bmap3.contourf(xv_new, yv_new, np.flipud(wsa), cmap='jet', latlon=True)
        plt.colorbar(cf3, orientation='vertical',cax=cax3, label='Wind Stress Anomaly (N/m^2)')
        axes[3].set_title('Wind Stress Anomaly')

        axes[3].set_xticks(np.arange(min_lon, max_lon, 2))
        axes[3].set_yticks(np.arange(min_lat, max_lat, 2))

        axes[3].set_xlabel('Longitude')
        axes[3].set_ylabel('Latitude')


        plt.savefig(save_plot_path / f'preprocess_pipeline_{region_path}_day_{i}.png', bbox_inches='tight')
        plt.close(fig)






def plot_trap_functions(trap_functions: list, cores: list, centroids: list,region_path: str, year: str, defuzzification_function:str, save_path: Path)->None:

    """Plots the trapezoidal functions and their centroids.
    @:param trap_functions: List of trapezoidal functions.
    @:param cores: List of core points for each trapezoidal function.
    @:param centroids: List of centroids for each trapezoidal function.
    @:param region_path: The path to the region.
    @:param year: The year for which the functions are plotted.
    @:param save_path: The path where the plot will be saved.
    @:return: None"""

    day_strings = generate_month_days()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['r','b','g','m','c','y']

    index = 0
    for t, cent, color, core in zip(trap_functions, centroids, colors, cores):
        index += 1
        ax.plot(t, label=f'USP-{index}',color=color)
        ax.plot(core[0], 1, '.', markersize=8, color=color)
        ax.text(core[0], 1.01, f'{day_strings[int(core[0])]}', color='black', fontsize=8, ha='center', va='bottom')
        ax.plot(core[1], 1, '.', markersize=8, color=color)

        if core[1] == 366:
            ax.text(core[1], 1.01, f'{day_strings[int(core[1])-1]}', color='black', fontsize=8, ha='center', va='bottom')
        else:
            ax.text(core[1], 1.01, f'{day_strings[int(core[1])]}', color='black', fontsize=8, ha='center', va='bottom')
        plt.vlines(cent, 0, 1, color=color, linestyle='--', label=f'Centroid: {day_strings[int(cent)]}')

    plt.xticks(np.arange(0, 365, step=30), day_strings[::30], rotation=45)
    plt.title(f'Trapezoidal Functions and Centroids-{region_path}-{year}')
    plt.legend(framealpha=1, fontsize='small', loc='center left')
    plt.savefig(save_path / f'trap_functions_{defuzzification_function}.png')
    plt.show()







def plot_decision_tree(dt: DecisionTreeClassifier,year: str,region: str, d: int, feature_names: [], class_names: [], mode: str, output_path: Path)->None:

    """Plots the decision tree model.
    @:param dt: The trained decision tree model.
    @:param year: The year for which the model was trained.
    @:param region: The region for which the model was trained.
    @:param d: The depth of the decision tree.
    @:param feature_names: List of feature names used in the model.
    @:param class_names: List of class names used in the model.
    @:param output_path: The path where the plot will be saved.
    @:return: None"""

    fig = plt.figure(figsize=(25,20))
    _ = plot_tree(dt, feature_names = feature_names,class_names=class_names,filled=True,precision=2)

    fig.savefig(output_path / f'{year}_{region}_{mode}_{str(d)}_decision_tree.png')
    #clear the figure to avoid overlap in subsequent plots
    plt.close(fig)




def plot_forest_feature_importances(rf, feature_names, year, region, dataset, output_path)->None:

    """Plots the random forest model's feature importances.
    @:param rf: The trained random forest model.
    @:param feature_names: List of feature names used in the model.
    @:param year: The year for which the model was trained.
    @:param region: The region for which the model was trained.
    @:param dataset: The dataset used for training the model.
    @:param output_path: The path where the plot will be saved.
    @:return: None"""

    feature_importances = rf.feature_importances_

    if dataset == 'ds1':
        feature_names = feature_names[:30]
    elif dataset == 'ds2':
        feature_names = feature_names[:34]
    else:
        feature_names = feature_names

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)

    # Plot the top 10 feature importances
    plt.figure(figsize=(10, 10))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title(f'Top 10 Feature Importances - {year} - {region} - {dataset}')
    plt.xticks(rotation=45)  # Rotate feature names for better readability
    plt.savefig(output_path / f'{year}_{region}_{dataset}_feature_importances.png')
    #plt.show()
    plt.close()