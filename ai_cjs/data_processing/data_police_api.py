import logging
import os
import pickle
import subprocess
import sys
from copy import copy
import datetime
import numpy as np
from OSGridConverter import latlong2grid, grid2latlong
import pandas as pd
import requests
from time import sleep, time

from ai_cjs.config import data_dir

logging.warning("New load attempt at: " + str(time()))
time_since_last_call = 0

staticmaps_api = "your-api-key"

def call_api(poly, date, crime_cat="all-crime", api_ret=[]):
    """
    Function to call the police API
    :param poly: the polygon (series of lat_long points) in which the crime is extracted
    :param date: the date string to call the api
    :param crime_cat: string to set the category of crimes to extract
    :param api_ret: list to store api returns (updated in place)
    """
    # create a variable to assert that we do not call the api too often
    global time_since_last_call
    sleep(np.max([1/4 - (time() - time_since_last_call)/4, 0]))  # limit to 2 calls per sec so not to max out
    # _call variable is used to progressively build the api call
    _call = "https://data.police.uk/api/crimes-street/" + crime_cat + "?"
    _call += "poly=" + ":".join("{:.4f}".format(lat_long.latitude) + "," + "{:.4f}".format(lat_long.longitude)
                                for lat_long in poly)
    _call += "&date=" + date
    logging.info(str(time()) + ": Calling PoliceAPI with: " + _call)
    # return the call object and assert it returned correctly
    _r = requests.get(_call)
    time_since_last_call = time()
    assert _r.status_code == 200, "Request failed with status code {}\n Request content:\n{}".format(_r.status_code,
                                                                                                     _r.content)
    # get the data from the call object and parse it to a pandas data frame
    _data = _r.json()
    _data_frame = pd.json_normalize(_data, sep="_")

    # these loops make calls to create google map api images (if data was returned) this can be safely removed and
    # replaced with 'return _data_frame'
    rows = []
    if _data_frame.size > 0:
        for lat_group in _data_frame.groupby(by=['location_latitude']):
            for long_group in pd.DataFrame(lat_group[1]).groupby(by=['location_longitude']):
                rows.append([len(long_group[1].index), lat_group[0], long_group[0]])
        google_map_area = "https://maps.googleapis.com/maps/api/staticmap?size=400x400&zoom=15&" + \
                          "path=color:black|weight:1|fillcolor:0xFFFF0033|" + \
                          "|".join("{:.4f}".format(lat_long.latitude) +
                                   "," +
                                   "{:.4f}".format(lat_long.longitude)
                                   for lat_long in poly) + "&" +\
                          "&".join("markers=size:mid%7Clabel:{}%7C".format(int(row[0])) +
                                   "{:.4f}".format(float(row[1])) +
                                   "," +
                                   "{:.4f}".format(float(row[2]))
                                   for row in rows) + "&" +\
                          "key={}".format(staticmaps_api)
        api_ret += \
            ["path=color:black|weight:2|fillcolor:0xFFFF0033|" +
             "|".join("{:.4f}".format(lat_long.latitude) +
                      "," + "{:.4f}".format(lat_long.longitude)
                      for lat_long in poly+[poly[0]]) + "&" +
             "&".join("markers=size:mid%7Clabel:{}%7C".format(int(row[0])) +
                      "{:.4f}".format(float(row[1])) +
                      "," +
                      "{:.4f}".format(float(row[2]))
                      for row in rows) +
             "&"]
        print(google_map_area)
        unwanted_cols = ["outcome_status", "location_type", "context", "persistent_id", "id", "location_subtype",
                         "location_street_id", "location_street_name", "outcome_status_category",
                         "outcome_status_date"]
        col_to_drop = set.intersection(set(unwanted_cols), set(_data_frame.head()))
        _data_frame.drop(columns=col_to_drop, inplace=True)
        return _data_frame
    else:
        print("No crimes_in", poly[0].latitude, poly[0].longitude)
        api_ret += \
            ["path=color:black|weight:2|fillcolor:0xFFFF0033|" +
             "|".join("{:.4f}".format(lat_long.latitude) +
                      "," + "{:.4f}".format(lat_long.longitude)
                      for lat_long in poly+[poly[0]]) +
             "&"]
        return _data_frame


def get_crimes_in_area(ll_origin, _side, _date=None, api_ret=[]):
    """
    Function to call crimes from the api
    :param ll_origin: tuple, the origin for crime extraction in latitude longitude
    :param _side: int, side length in meters
    :param _date: tuple, month year
    :param api_ret: list to append the api call to (changed inplace)
    """
    # assign default date if necessary
    if _date is None:
        _today = datetime.date.today()
        if _today.month == 1:
            _date_str = str(_today.year-1) + "-12"
        else:
            _date_str = "{:d}-{:02d}".format(_today.year, _today.month-1)
    else:
        print(_date)
        _date_str = "{:d}-{:02d}".format(_date[1], _date[0])

    # check if the latlong poly needs creating
    if len(ll_origin) >= 4:
        ll_corners = ll_origin
    else:
        grid_corners = [latlong2grid(ll_origin[0], ll_origin[1]) for _ix in range(4)]

        grid_corners[1].E += 0; grid_corners[1].N += _side
        grid_corners[2].E += _side; grid_corners[2].N += _side
        grid_corners[3].E += _side; grid_corners[3].N += 0
        ll_corners = [grid2latlong(str(corner)) for corner in grid_corners]
    # call the api and return the dataframe of crimes
    _crimes = call_api(ll_corners, _date_str, api_ret=api_ret)
    return _crimes


def add_ne(osgridref_ob, x, y):
    """
    Add to nothings and eastings
    :param osgridref_ob: osgridref object
    :param x: x value to add to the easting
    :param y: y value to add to the northing
    """
    osgridref_ob.E += x
    osgridref_ob.N += y
    return osgridref_ob


def get_grid_profile(_grid_origin, _grid_n, _sub_grid_size, _name="", _month=(1, 2020)):
    """
    Function to return crimes over a grid with optional API call made to make an image from google maps.
    :param _grid_origin: the origin (lower left point) of the grid in latitude longitude
    :param _grid_n: the number of cells along one side of the grid.
    :param _sub_grid_size: the length of the side of one of the cells that make up the sub grid.
    :param _name: optional name parameter used for a descriptive save file
    :param _month: the month to extract crimes from.
    :return: crimes, a list of the crimes extracted in each grid cell.
    :return: crime_count, a numpy array shaped to the grid with the crime count in each cell.
    """
    # get origin in northings eastings
    ll_origin = latlong2grid(_grid_origin[0], _grid_origin[1])
    # make array of subgrid origins in northings and eastings
    x = np.arange(0, (_grid_n+1)*_sub_grid_size, _sub_grid_size)
    y = np.arange(0, (_grid_n+1)*_sub_grid_size, _sub_grid_size)
    # start building the google maps api call
    api_call = ["https://maps.googleapis.com/maps/api/staticmap?size=400x400&zoom=15&"]
    # turn the grid into latlong coords
    ll_origins = [[grid2latlong(str(add_ne(copy(ll_origin), _x, _y))) for _x in x] for _y in y]
    # return the crimes in each grid (api_call is updated as we go)
    _crimes = [[get_crimes_in_area([ll_origins[_ix][_iy],
                                    ll_origins[_ix+1][_iy],
                                    ll_origins[_ix+1][_iy+1],
                                    ll_origins[_ix][_iy+1]],
                                   _sub_grid_size, _month, api_ret=api_call)
                for _ix in range(_grid_n)]
               for _iy in range(_grid_n)]
    # count the crimes in each cell
    crime_count = np.array([[len(_crimes[_ix][_iy].index) for _ix in range(_grid_n)] for _iy in range(_grid_n)])
    # finish the api call with the key
    api_call += ["key={}".format(staticmaps_api)]
    api_call = "".join(api_call)
    # check if the API key is valid (max character length is 8192)
    if len(api_call) > 8192:
        # group the crimes by cell
        api_call = "https://maps.googleapis.com/maps/api/staticmap?size=400x400&zoom=15" + "&" + \
                   "path=color:black|weight:2|fillcolor:0xFFFF0033|" +\
                   "|".join([
                             "|".join(["{:.4f},{:.4f}".format(lat_long.latitude, lat_long.longitude)
                                       for lat_long in square
                                       ])
                             for square in [[ll_origins[_ix][0],
                                             ll_origins[_ix][_iy], ll_origins[_ix+1][_iy],
                                             ll_origins[_ix+1][_iy+1], ll_origins[_ix][_iy+1],
                                             ll_origins[_ix][_iy],
                                             ll_origins[0][_iy], ll_origins[0][0]]
                                            for _ix in range(_grid_n)
                                            for _iy in range(_grid_n)]

                             ]) + "&" +\
                   "&".join(["markers=label:{}%7C".format(crime_count[_ix][_iy]) +
                             "{:.4f}".format((ll_origins[_ix][_iy].latitude + ll_origins[_ix+1][_iy+1].latitude)/2) +
                             "," +
                             "{:.4f}".format((ll_origins[_ix][_iy].longitude + ll_origins[_ix+1][_iy+1].longitude)/2)
                             for _ix in range(_grid_n)
                             for _iy in range(_grid_n)]) +\
                   "&key={}".format(staticmaps_api)
    # check the api_call length again, if it is too long print an error
    if len(api_call) > 8192:
        print("api_call too long, chars:{}".format(len(api_call)))
    else:
        # show the api_call
        print(api_call)
    return _crimes, crime_count


def defined_call(lat_long=(51.5081, -0.1469), _mm_yyyy=None, side_n=10, side_l=150):
    """
    Function to make a series of calls on the API
    :param lat_long: tuple (float, float), the origin of the grid to call in latitude longitude
    :param _mm_yyyy: list [tuple (month, year), ...], months to extract crimes over
    :param side_n: int, the number of cells per side of the grid (n squared if the total number of cells)
    :param side_l: int, number of meters for a grid side
    :return: crime_database, grid_profile, pandas database of the crimes, numpy 2d grid
    """
    grid_profile = []
    crime_database = []

    if _mm_yyyy is None:
        # use 2019
        _mm_yyyy = [(int(_month), int(2019)) for _month in range(1, 13, 1)]

    # for effiency check if this data exists
    for mth_yr in _mm_yyyy:
        element_path = "real_crime_profiles/ll_{}_{}_n_{}_l_{}/{}_{}/".format(lat_long[0], lat_long[1],
                                                                              side_n, side_l,
                                                                              mth_yr[0], mth_yr[1]).replace(" ", "")
        # check the path exists if so load the grid and the crime
        if os.path.exists(data_dir + element_path):
            logging.info('Loading from ' + element_path)
            crime_count_grid = np.load(data_dir + element_path+"grid_profile.npy")
            crime_db = pd.read_csv(data_dir + element_path + "crime_db.csv")
        # otherwise get the crimes
        else:
            print(mth_yr)
            crime_db, crime_count_grid = get_grid_profile(lat_long, side_n, side_l, _name="", _month=mth_yr)
            subprocess.call(["mkdir", "-p", data_dir + element_path])
            np.save(data_dir + element_path + "grid_profile.npy", crime_count_grid)
            crime_db = pd.concat(db for listB in crime_db for db in listB)
            crime_db.to_csv(data_dir + element_path + "crime_db.csv")
        # append to the databases
        crime_database.append(crime_db)
        grid_profile.append(crime_count_grid)
    return pd.concat(crime_database), np.stack(grid_profile)


if __name__ == "__main__":
    # If script is run then get input from user or argv (commandline inputs)
    if len(sys.argv) == 1:
        _lat = input("Input latitude")
        _long = input("Input longitude")
        years = input("Please enter year or comma separated list of years (no whitespace).\n").split(sep=",")
        months = input("Please enter year or comma separated list of months (no whitespace).\n"
                       "Months will be applied to all years.\n"
                       "Months in the current year will be truncated to latest data.\n").split(sep=",")
        grid_n = input("Input number of cells\n")
    else:
        months = sys.argv[1].split(sep=",")
        years = sys.argv[2].split(sep=",")
        _lat, _long = str.split(sys.argv[3], sep=",")
        cores = int(sys.argv[4])
        side = int(sys.argv[5])
        grid_n = np.sqrt(cores) * side
    # set up the months and years in the correct format and check validity
    mm_yyyy = []
    today = datetime.date.today()
    for year in years:
        assert today.year > int(year), "Year, {}, is beyond current date".format(int(year))
        for month in months:
            if today.year == int(year) and today.month < int(month):
                print("Month {} in Year, {}, is beyond current date".format(int(month), int(year)))
            else:
                mm_yyyy.append((int(month), int(year)))

    # set up the lat long tuples with default (when required) value and proper formatting
    if (_lat or _long) == "":
        _lat, _long = (51.5081, -0.1469)  # Somewhere in london

    _grid_origin = (float(_lat), float(_long))

    # make the path to the place where the data is stored
    add_path_llmy = ""
    add_path_llmy += "lat_long_{}_{}/".format(_lat, _long)
    months = sorted(months)
    switch = False
    for i, month in enumerate(months):
        if ~switch:
            add_path_llmy += str(month)
            switch = True
            last_month = month
        elif month - 1 == months[i - 1]:
            if i + 1 == len(months):
                add_path_llmy += "-" + str(month)  # this month is the last month
            else:
                pass  # this month is the next month in sequence
        else:
            if last_month != months[i - 1]:
                add_path_llmy += "-" + str(months[i - 1])
            add_path_llmy += "_" + str(month)
            last_month = month

    years = sorted(years)
    switch = False
    for i, year in enumerate(years):
        if ~switch:
            add_path_llmy += str(year)
            switch = True
            last_year = year
        elif year - 1 == years[i - 1]:
            if i + 1 == len(years):
                add_path_llmy += "-" + str(year)  # this year is the last year
            else:
                pass  # this year is the next year in sequence
        else:
            if last_year != years[i - 1]:
                add_path_llmy += "-" + str(years[i - 1])
            add_path_llmy += "_" + str(year)
            last_year = year
    # use the path created above to make a directory to store the outputs.
    subprocess.call(["mkdir", "-p", data_dir + "real_crime_profiles/{}/".format(add_path_llmy).replace(" ", "")])

    # make the api call using the data defined
    crimes, counts = defined_call(lat_long=_grid_origin, _mm_yyyy=mm_yyyy, side_n=int(grid_n))

    # save the data to the location made above
    np.save(data_dir + "real_crime_profiles/{}/grid_profile.npy".format(add_path_llmy).replace(" ", ""), counts)
    with open(data_dir + "real_crime_profiles/{}/pol_db.pkl".format(add_path_llmy).replace(" ", ""), 'wb') as f:
        pickle.dump(crimes, f)
