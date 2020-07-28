import functools
from os import walk
import logging
import pickle
import subprocess
import types

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ai_cjs import cte
from ai_cjs.config import figure_dir, data_dir, local_scratch
from ai_cjs.config import config_dict as _config_dict


def analyse(_data_dict, _key_higherichy=[], get_config=True, extra_path=[], large_files=False):
    """
    :param _data_dict: A set of nested dictionaries containing the model
                       outputs under keys matching the file paths of the data
    :param _key_higherichy: An orderd list of the keys relevent to get to the data.
    :param get_config: Set to true if running post simulation to load the config dict
    :param extra_path: list, extra strings appended to the filepaths to load the config from
    :param large_files: bool, set to true to call the functions that then load the data locations
    """

    if get_config:
        config_path = local_scratch
        for _key in extra_path+_key_higherichy[:-2]:
            config_path += _key + "/"
        with open(config_path + ".param_pkl.pkl", "rb") as f:
            config_dict = pickle.load(f)
    else:
        config_dict = _config_dict

    out_file = figure_dir
    # Add the path for the given data_dict
    for _key in _key_higherichy:
        out_file += _key + "/"
        _data_dict = _data_dict[_key]
    subprocess.call(["mkdir", "-p", out_file])

    # load data from MPI output
    if large_files:
        total_crime_history = _data_dict['ch']()
        total_crime_reports = _data_dict['cr']()
        total_cond_intn = _data_dict['ci']()
    else:
        total_crime_history = _data_dict['ch']
        total_crime_reports = _data_dict['cr']
        total_cond_intn = _data_dict['ci']

    # cast as pandas frames with correct column names and data types
    total_crime_history = pd.DataFrame(total_crime_history, columns=['grid_num', 'time'])
    total_crime_reports = pd.DataFrame(total_crime_reports, columns=['time', 'grid_num'])
    total_crime_history['grid_num'] = total_crime_history['grid_num'].astype(np.int)
    total_crime_reports['grid_num'] = total_crime_reports['grid_num'].astype(np.int)

    # get the size of the total grid
    n = np.sqrt(config_dict['Cores']).astype(int)*config_dict['SubGridN']
    u_grid = np.zeros((n, n))

    time_frame = 1
    start_time = config_dict['InitDays']
    # Loop over each conditional intensity step
    silhouette_avgs = []
    for i in range(1, len(total_cond_intn) - 1):
        # Plot the kmeans clustering
        silhouette_avg = cte.kmeans_cluster_plot(u_grid, total_crime_reports,
                                                 total_crime_history, time_frame, start_time + i,
                                                 'kmean_cluster_{:02d}'.format(i), fig_loc=out_file, make_plot=False)
        silhouette_avgs.append(silhouette_avg)
    plt.plot(np.arange(1, len(silhouette_avgs) + 1), silhouette_avgs)
    plt.xlabel("Days")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(out_file + "silhouette_score.png")
    plt.close()
    np.save(out_file+"silhouette_score.npy", silhouette_avgs)
    return np.array(silhouette_avgs)


def plot(_data_dict, _key_higherichy=[], get_config=True, extra_path=[], large_files=False):
    """
    :param _data_dict: A set of nested dictionaries containing the model
                       outputs under keys matching the file paths of the data
    :param _key_higherichy: An orderd list of the keys relevent to get to the data.
    :param get_config: Set to true if running post simulation to load the config dict
    :param extra_path: list, extra strings appended to the filepaths to load the config from
    :param large_files: bool, set to true to call the functions that then load the data locations
    """

    if get_config:
        config_path = local_scratch
        for _key in extra_path+_key_higherichy[:-2]:
            config_path += _key + "/"
        with open(config_path + ".param_pkl.pkl", "rb") as f:
            config_dict = pickle.load(f)
    else:
        config_dict = _config_dict

    out_file = figure_dir
    # Add the path for the given data_dict
    for _key in _key_higherichy:
        out_file += _key + "/"
        _data_dict = _data_dict[_key]
    subprocess.call(["mkdir", "-p", out_file])

    # load data from MPI output
    if large_files:
        total_crime_history = _data_dict['ch']()
        total_crime_reports = _data_dict['cr']()
        total_cond_intn = _data_dict['ci']()
    else:
        total_crime_history = _data_dict['ch']
        total_crime_reports = _data_dict['cr']
        total_cond_intn = _data_dict['ci']

    # cast as pandas frames with correct column names and data types
    total_crime_history = pd.DataFrame(total_crime_history, columns=['grid_num', 'time'])
    total_crime_reports = pd.DataFrame(total_crime_reports, columns=['time', 'grid_num'])
    total_crime_history['grid_num'] = total_crime_history['grid_num'].astype(np.int)
    total_crime_reports['grid_num'] = total_crime_reports['grid_num'].astype(np.int)

    # get the size of the total grid
    n = np.sqrt(config_dict['Cores']).astype(int)*config_dict['SubGridN']
    u_grid = np.zeros((n, n))

    time_frame = 1
    start_time = config_dict['InitDays']

    # get a mask for any cells that start as outliers
    mean = np.mean(total_cond_intn[1])
    std = np.std(total_cond_intn[1])
    outlier_mask = total_cond_intn[1] > mean + std
    # Loop over each conditional intensity step and make plots
    silhouette_avgs = []
    for i in range(1, len(total_cond_intn) - 1):
        # Get the figure to pass to the plotting functions
        _fig, _ax = plt.subplots()
        temp_fig = [_fig, _ax]
        # Plot the crime profile for the real crime
        temp_fig = cte.plot_crime_profile(u_grid, total_crime_history, time_frame, start_time + i,
                                          'tc_test{:02d}'.format(i), return_fig=True, fig_ax=temp_fig,
                                          fig_loc=out_file)
        # Plot the predpol contours ontop the crimes and save the figure (then close)
        cte.plot_predpol(total_cond_intn[i].reshape((n, n)), 'tc_test{:02d}'.format(i),
                         fig_ax=temp_fig, fig_loc=out_file)
        plt.close()

        # Get the figure to pass to the plotting functions
        _fig, _ax = plt.subplots()
        temp_fig = [_fig, _ax]
        # Plot the crime profile for the recorded crime
        temp_fig = cte.plot_crime_profile(u_grid, total_crime_reports,
                                          time_frame, start_time + i, 'rc_test{:02d}'.format(i), return_fig=True,
                                          fig_ax=temp_fig, fig_loc=out_file)
        # Plot the predpol contours ontop the crimes and save the figure (then close)
        cte.plot_predpol(total_cond_intn[i].reshape((n, n)), 'rc_test{:02d}'.format(i),
                         fig_ax=temp_fig, fig_loc=out_file)
        plt.close()

        # Make three corner plots showing the correlations between pred pol, true crime and reported crime
        # Get the figure to pass to the plotting functions
        fig_axs = plt.subplots(2, 2)
        # Plot for all data
        fig_axs = cte.correlation_function(u_grid, total_cond_intn[i + 1], total_crime_reports,
                                           total_crime_history, time_frame, start_time + i,
                                           'correlation_{:02d}'.format(i),
                                           return_fig=True, fig_axs=fig_axs, fig_loc=out_file)
        # Plot for the data that started above the mean save figure (then close)
        cte.correlation_function(u_grid, total_cond_intn[i + 1], total_crime_reports,
                                 total_crime_history, time_frame, start_time + i,
                                 'correlation_{:02d}'.format(i),
                                 hysteresis=outlier_mask, return_fig=False, fig_axs=fig_axs, fig_loc=out_file)
        plt.close()

        # Plot the kmeans clustering
        silhouette_avg = cte.kmeans_cluster_plot(u_grid, total_crime_reports,
                                                 total_crime_history, time_frame, start_time + i,
                                                 'kmean_cluster_{:02d}'.format(i), fig_loc=out_file)
        plt.close()
        silhouette_avgs.append(silhouette_avg)
    plt.plot(np.arange(1, len(silhouette_avgs)+1), silhouette_avgs)
    plt.xlabel("Days")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(out_file + "silhouette_score.png")
    plt.close()


def load_data(_fp):
    """
    Load known data, pass unwanted files
    :param _fp: str, path to data file
    :return: Data from known file types, None for known but unwanted file types, Error for unknown file types
    """
    # Get the file extension from the path
    file_type = _fp.split("/")[-1].split(".")[-1]
    # Switch statements that load file if appropriate, pass file if not wanted or print a warning that the file type
    # was unknown.
    if file_type == "npy":
        return np.load(_fp, allow_pickle=True)
    elif file_type == "pkl":
        with open(_fp, "rb") as _file:
            return pickle.load(_file)
    elif file_type in ["png", "log", "txt"]:
        pass
    else:
        print("File Type: .", file_type, " not known")


def descend(_dict, _path, _call=0, _data=None):
    """
    Recursively descend a file path creating new dictionaries until the base is reached. If the file path has data
    associated with it add the data to the bottom of the nested dictionaries.
    :param _dict: dict, The current dictionary
    :param _path: str, The current path
    :param _call: int, the number of times function has been called recursively
    :param _data: various, data to assign to dict
    :return: dict or None if data is assigned
    """
    # If the levels of decent match the path length (i.e. function has reached the lowest directory in this path)
    if _call+1 == len(_path):
        # If there is data create a key and assign the data
        if _data is not None:
            ft_len = len(_path[_call].split(".")[-1]) + 1
            _dict[_path[_call][:-ft_len]] = _data
        # else make an empty dict
        else:
            _dict[_path[_call]] = {}
            return _dict
    else:
        # get the next level down increase call and recur this function
        new_dict = _dict.get(_path[_call])
        _call += 1
        descend(new_dict, _path, _call, _data)


def descend_and_plot(_dict, large_files, _dict_orig=None, key_path=[], data_paths=None):
    """
    Recursively descend a file path creating new dictionaries until the base is reached. If the file path has data
    associated with it add the data to the bottom of the nested dictionaries.
    :param _dict: dict, The current dictionary
    :param _dict_orig: dict, The original dictionary
    :param key_path: list[str], Recursively built list of dictionary keys for plotting
    :param large_files: bool, toggle for if we load the data or a functools partial object
    """
    # if this is the first call then assign the dict as the original dictionary
    if _dict_orig is None:
        _dict_orig = _dict
    # get the keys for the current dictionary
    _keys = _dict.keys()
    # set the flag for no decent to true
    no_dict = True
    for _key in _keys:
        #  check next layer is also a dictionary
        if type(_dict[_key]) == dict:
            descend_and_plot(_dict[_key], large_files, _dict_orig=_dict_orig, key_path=key_path + [_key],
                             data_paths=data_paths)
            no_dict = False  # Flag that there is a dict on this level
    # at each lowest level attempt to plot
    if no_dict and (key_path in data_paths):
        try:
            print("Plotting")
            print("key_path", key_path)
            plot(_dict_orig, key_path, large_files=large_files)
        except KeyError as e:
            print('Error: No Key:', e, key_path)


def descend_and_analyse(_dict, large_files, _dict_orig=None, key_path=[], data_paths=None, stacked_silhouette=None,
                        parameters=[], initial_call=True):
    """
    Recursively descend a file path creating new dictionaries until the base is reached. If the file path has data
    associated with it add the data to the bottom of the nested dictionaries.
    :param _dict: dict, The current dictionary
    :param _dict_orig: dict, The original dictionary
    :param key_path: list[str], Recursively built list of dictionary keys for plotting
    :param large_files: bool, toggle for if we load the data or a functools partial object
    :param data_paths: list, paths where descend has found valid data
    :param stacked_silhouette: numpy array, used to collect the silhouettes from the recursive search
    :param parameters: list, list to collect the model config for each silhouette from the recursive search
    :param initial_call: bool, set to true for the top level call recursive calls have this parameter as false
    :return: stacked_silhouette, parameters; the data collected recursively saved by the initial call
    """
    # if this is the first call then assign the dict as the original dictionary
    if _dict_orig is None:
        _dict_orig = _dict
    # get the keys for the current dictionary
    _keys = _dict.keys()
    # set the flag for no decent to true
    no_dict = True
    for _key in _keys:
        #  check next layer is also a dictionary
        if type(_dict[_key]) == dict:
            stacked_silhouette, parameters = descend_and_analyse(_dict[_key], large_files, _dict_orig=_dict_orig,
                                                                 key_path=key_path + [_key], data_paths=data_paths,
                                                                 stacked_silhouette=stacked_silhouette,
                                                                 parameters=parameters, initial_call=False)
            no_dict = False  # Flag that there is a dict on this level
    # at each lowest level attempt to plot
    if no_dict and (key_path in data_paths):
        try:
            print("analyse")
            print("key_path", key_path)
            if stacked_silhouette is not None:
                stacked_silhouette_temp = analyse(_dict_orig, key_path, large_files=large_files)
                stacked_silhouette = np.vstack((stacked_silhouette, stacked_silhouette_temp))
            else:
                stacked_silhouette = analyse(_dict_orig, key_path, large_files=large_files)
            parameters.append(key_path)

        except KeyError as e:
            print('Error: No Key:', e, key_path)

    if initial_call:
        with open("ParameterList.txt", "w") as f:
            for line in parameters:
                f.write(str(line) + "\n")
        np.save("stacked_silhouette.npy", stacked_silhouette)
    else:
        return stacked_silhouette, parameters


def return_call(func):
    """
    Modifies a function to instead return a defined call to the function to prevent overloading memory
    """
    def inner(*args):
        return functools.partial(func, *args)
    return inner


def walk_dirs(dirs_to_walk=[""], make_plot=True, analyse_sil=False, _local_scratch=local_scratch, large_files=False):
    """
    From a start directory create a dictionary that mimics the descendant file structure loading files where possible
    then probe this structure to make plots
    :param dirs_to_walk: list[str]; list of directories to walk
    :param make_plot: bool; flag to trigger plotting
    :param _local_scratch: str; file path for the local scratch file in which to look for the directories; default,
    :param large_files: bool; if True load_data is modified to return a function call in place of the actual data
    local_scratch from config.py
    """

    if large_files:
        _load_data = return_call(load_data)
    else:
        _load_data = load_data

    data_paths = []  # store the paths to any folder containing data
    count = [0, 0, 0]  # count the number of 'roots', directories, and files processed
    for dir_to_walk in dirs_to_walk:
        print("Walking: ", _local_scratch+dir_to_walk)
        # create a clean dict to operate on
        data_dict = {}
        temp_data_paths = []
        # use walk to traverse file structures
        for root_str, dirs, files in walk(_local_scratch+dir_to_walk):
            root = root_str[len(_local_scratch+dir_to_walk):]
            # filter root directory and hidden directories
            if len(root) == 0:
                continue
            if root[0] == ".":
                continue
            # split the root on forward slash
            root = root.split(sep="/")
            temp_dict = data_dict
            count[1] = 0
            count[2] = 0
            # for each folder on the path create or descend into the dictionary
            for folder in root:
                count[1] += 1
                print(count, end="\r")
                if folder not in temp_dict.keys():
                    temp_dict = descend(data_dict, root)
                else:
                    temp_dict = temp_dict[folder]
            # for each file in the path load it then descend and place it in the dictionary
            for file in files:
                count[2] += 1
                print(count, end="\r")
                descend(data_dict, root + [file], _data=_load_data(root_str + "/" + file))
            # using a known data file name add the data locations
            if "cr.npy" in files:
                temp_data_paths.append(root)
                data_paths.append(root)
            print(count, end="\r")
            count[0] += 1
        # either make the plots or
        if make_plot:
            descend_and_plot(data_dict, large_files, data_paths=temp_data_paths)
        elif analyse_sil:
            descend_and_analyse(data_dict, large_files, data_paths=temp_data_paths)
        else:
            return data_dict, data_paths


if __name__ == "__main__":
    walk_dirs()
