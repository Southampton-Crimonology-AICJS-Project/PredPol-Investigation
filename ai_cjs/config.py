#  This script defines the paths used for data input/output and scratch storage
#  Each set of path variables should be block commented and labelled.
#  After import this script allows each other script to inherit the configuration dictionary and the file paths
import datetime
import getopt
import logging
import numpy as np
import pickle
from scipy import stats
import subprocess
import sys
import pathlib


config_path = str(pathlib.Path(__file__).parent.absolute())

with open(config_path+"/who_am_i.txt", "r") as f:
    for i, line in enumerate(f):
        if i == 1:
            if line[-1:] == "\n":
                top_scratch_dir = line[:-1]
            else:
                top_scratch_dir = line

            if line[-1] != "/":
                line += "/"

# Runs on script import
if str.split(sys.argv[0], sep="/")[-1] in {"config.py", "crim_theory_mpi.py", "gather_plot_mpi.py"} \
        and (len(sys.argv) > 1):
    # time conversions and definitions
    month = 31
    day = 1

    # Core Arguments:
    #   # 0: Script_name,
    #   # 1: time to run in days,
    #   # 2: sub grid n
    #   # 3: init_model
    #   # 4: daily_model
    #   # 5: init_crime
    #   # 6: daily_crime
    #   # 7: conv_threshold
    #   # 8: conv_limit
    #   # 9: cores
    #   #-1: flag
    # Optional Arguments:
    #   #--BaseReport: the base reporting level
    #   #--RepEn: reporting_enhance
    #   #--EnLvl: Modify to the enhancement model. For Factor this is the factor, for proportional this is the max
    #   #--FixPer percentage of crime to be seeded in the fixed locations
    # assign args
    system_arguments = sys.argv[:10]
    # split args
    # time to run
    days = int(system_arguments[1])
    # grid_sizes
    sub_grid_n = int(system_arguments[2])

    # number of crimes across the grid and crime model
    init_model = str(system_arguments[3])
    daily_model = str(system_arguments[4])

    init_crime = int(system_arguments[5])
    daily_crime = int(system_arguments[6])

    # number of days to spread the init crime over
    init_days = 5
    # Convergence limit/threshold for predpol algorithm
    conv_threshold = float(system_arguments[7])  # get to the point where nothing changes below this threshold
    conv_lim = int(system_arguments[8])  # break after this many steps

    cores = int(sys.argv[9])

    # make a configuration dictionary to be used by the rest of the codes
    config_dict = {
        'Days': int(days),
        'InitDays': int(init_days),
        'SubGridN': int(sub_grid_n),
        'Cores': int(cores),
        'InitModel': str(init_model),
        'DailyModel': str(daily_model),
        'InitCrime': int(init_crime),
        'DailyCrime': int(daily_crime),
        'ConvThresh': float(conv_threshold),
        'ConvLim': int(conv_lim),
        'BaseRep': float(0.333),
        'RepEn': [None, None, None],
        'FixedPer': None,
        'RealFake': [None, None]
    }

    # additional path for storing multiple models
    add_path = ""
    add_path += init_model + "/"
    add_path += daily_model + "/"

    # prime variables for system argument option appending
    add_path_re = ""
    add_path_relv = ""
    add_path_fix = ""
    add_path_latlong = ""
    add_path_my = ""
    add_path_llmy = ""
    report_enhance = False
    enhance_lvl = None
    fixed_percentage = None
    # Split the remaining argument using the getopt parser
    opts, args = getopt.getopt(sys.argv[10:], "", ["BaseReport=", "RepEn", "RepEnMdl=", "EnLvl=", "FixPer=",
                                                   "Months=", "Years=", "Lat_Long="])
    # loop over the optional arguments and set paths/variables accordingly
    for opt, arg in opts:
        if opt in ("--BaseReport",):
            add_path_re += arg + "/"
            assert 0 <= float(arg)/100 <= 1, "BaseReport must be in [0,100] was {}".format(float(arg))
            config_dict['BaseRep'] = float(arg)/100
        if opt in ("--RepEn",):
            report_enhance = True
            add_path_re += "report_enhance/"
            config_dict['RepEn'][0] = True
        if opt in ("--RepEnMdl",):
            report_enhance = True
            add_path_re += arg + "/"
            config_dict['RepEn'][1] = str(arg)
        elif opt in ("--EnLvl",):
            enhance_lvl = arg
            add_path_relv += arg + "/"
            config_dict['RepEn'][2] = float(arg)/100
        elif opt in ("--FixPer",):
            fixed_percentage = arg
            add_path_fix += "fix_per_" + arg + "/"
            config_dict['FixedPer'] = float(arg)
        if opt in ("--Lat_Long",):
            lat_long = str.split(arg, ",")
            add_path_latlong += "lat_long_{}_{}/".format(lat_long[0], lat_long[1])
        if opt in ("--Months",):
            months = sorted([int(i) for i in arg.split(sep=",")])
            switch = False
            for i, month in enumerate(months):
                if ~switch:
                    add_path_my += str(month)
                    switch = True
                    last_month = month
                elif month - 1 == months[i - 1]:
                    if i + 1 == len(months):
                        add_path_my += "-" + str(month)  # this month is the last month
                    else:
                        pass  # this month is the next month in sequence
                else:
                    if last_month != months[i - 1]:
                        add_path_my += "-" + str(months[i - 1])
                    add_path_my += "_" + str(month)
                    last_month = month
        if opt in ("--Years",):
            years = sorted([int(i) for i in arg.split(sep=",")])
            switch = False
            for i, year in enumerate(years):
                if ~switch:
                    add_path_my += str(year)
                    switch = True
                    last_year = year
                elif year-1 == years[i-1]:
                    if i + 1 == len(years):
                        add_path_my += "-" + str(year)  # this year is the last year
                    else:
                        pass  # this year is the next year in sequence
                else:
                    if last_year != years[i-1]:
                        add_path_my += "-" + str(years[i-1])
                    add_path_my += "_" + str(year)
                    last_year = year
            add_path_my += "/"
    if {"--Lat_Long", "--Years", "--Months"}.issubset(sys.argv[10:]):
        today = datetime.date.today()
        mm_yyyy = []
        for year in years:
            assert today.year > int(year), "Year, {}, is beyond current date".format(int(year))
            for month in months:
                if today.year == int(year) and today.month < int(month):
                    pass
                else:
                    mm_yyyy.append((int(month), int(year)))
        add_path_llmy += add_path_latlong + add_path_my
        counts = np.load("{}.__scratch__/data/".format(top_scratch_dir).replace(" ", "") +
                         "real_crime_profiles/{}/grid_profile.npy".format(add_path_llmy).replace(" ", ""))
        mean = counts.mean(axis=0)
        std = counts.std(axis=0)
        mean_std_month = []
        for i in counts:
            mean_std_month.append([i.mean(), i.std()])
        m, c = np.polyfit(mean.flatten(), std.flatten(), 1)
        assert m >= 0.0
        assert c >= 0.0
        kde = stats.gaussian_kde(mean.flatten())
        full_box_side = sub_grid_n*np.sqrt(cores)

        grid_mean = kde.resample(int(full_box_side**2)).flatten()
        grid_mean[grid_mean < 0] = 0
        grid_std = grid_mean*m + c
        config_dict['RealFake'] = [grid_mean, grid_std]

    add_path += add_path_latlong + add_path_my + add_path_fix + add_path_re + add_path_relv

    # Additional path for storing technical run data
    add_path += str(cores) + "_" + str(sub_grid_n) + "/"
    add_path += str(init_crime) + "_" + str(daily_crime) + "/"
    add_path += str(init_days) + "_" + str(days) + "/"

    # file locations
    scratch_dir = top_scratch_dir + add_path
    figure_dir = scratch_dir + "figures/"
    data_dir = scratch_dir + "data/"
    local_scratch = scratch_dir + "scratch/"

    if __name__ == "__main__":
        # Ensure these directories exist on the system
        subprocess.call(["mkdir", "-p", figure_dir])
        subprocess.call(["mkdir", "-p", data_dir])
        subprocess.call(["mkdir", "-p", local_scratch])
        subprocess.call(["mkdir", "-p", local_scratch+"mpi_out/"])
        subprocess.call(["mkdir", "-p", figure_dir+"crim_theory_eff/"])
        with open(top_scratch_dir + ".add_path.txt", "w") as f:
            f.write(add_path)
        with open(scratch_dir + ".param_pkl.pkl", "wb") as f:
            pickle.dump(config_dict, f)

    logging.basicConfig(filename=local_scratch + '{}.log'.format(str.split(sys.argv[0], sep="/")[-1][:-3]),
                        level=logging.DEBUG)
    logging.info("config dict:\n"+str(config_dict))
else:
    print("\nSaving all data to '{}.__scratch__/'".format(top_scratch_dir))
    subprocess.call(["mkdir", "-p", "{}.__scratch__/".format(top_scratch_dir)])
    figure_dir = "{}.__scratch__/figures/".format(top_scratch_dir)
    data_dir = "{}.__scratch__/data/".format(top_scratch_dir)
    local_scratch = "{}.__scratch__/local_scratch/".format(top_scratch_dir)
    config_dict = {}
