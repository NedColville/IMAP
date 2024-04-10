import numpy as np
from datetime import datetime

def alignInterp(datasets, num_points):
    """
    Align time series within multiple datasets and interpolate data for each dataset to the specified number of points.

    Parameters:
    datasets (list of lists): List containing time series and data arrays for multiple datasets.
    num_points (int): The desired number of points for the interpolated time series.

    Returns:
    interpolated_time_series (list of datetime.datetime): The interpolated time series for all datasets.
    interpolated_data (list of numpy.ndarray): Interpolated data arrays for each dataset in datasets.
    """
    # Convert datetime objects to Unix timestamp

    # Extract time series and data arrays for each dataset
    time_series = [dataset[0] for dataset in datasets]
    data_arrays = [dataset[1:] for dataset in datasets]
    time_series = [[datetime.timestamp(ts) for ts in ts_list] for ts_list in time_series]

    # Find the common start and end times for all datasets
    common_start_time = max([ts[0] for ts in time_series])
    common_end_time = min([ts[-1] for ts in time_series])

    # Create the common time series within the overlapping time range
    interpolated_time_series = np.linspace(common_start_time, common_end_time, num_points)

    # Interpolate data for each dataset
    interpolated_data = []
    for ts, data_array in zip(time_series, data_arrays):
        interpolated_data.append([np.interp(interpolated_time_series, ts, data) for data in data_array])

    # Convert Unix timestamps back to datetime objects
    interpolated_time_series = [datetime.fromtimestamp(ts) for ts in interpolated_time_series]

    return interpolated_time_series, np.array(interpolated_data)
