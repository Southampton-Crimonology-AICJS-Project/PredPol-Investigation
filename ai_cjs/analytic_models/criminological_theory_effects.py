import logging

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ai_cjs.config import figure_dir, config_dict


def get_crime_reports(_grid, _crime, mean_std=(0., 0.)):
    """
    Return a numpy array of 'reported crimes', created by sampling the 'total crime' by 1/3 plus some enhancement
    from the number of police dispatched by PredPol controlled by the model defined in config_dict['RepEn'].
    :param _grid: numpy array, A flattened grid of conditional intensities
    :param _crime: the crimes to randomly record/not report
    :param mean_std: tuple, the standard dev and mean of _grid
    :return: (N, 2) array of recorded crimes
    :return: report_threshold the thresholds used for crime reporting
    """

    logging.info("GetCrimeReports:\n")
    logging.debug("Grid:\n{}\nCrime:\n{}\nMeanStd:\n{}\n".format(_grid, _crime, (0., 0.)))

    _grid = np.nan_to_num(_grid)  # remove nan values from the grid
    report_threshold = np.full_like(_grid, config_dict['BaseRep'])  # make an array to store report_threshold
    # If the report enhancement is on
    if config_dict['RepEn'][0]:
        # Where there is a simple factor increase
        if config_dict['RepEn'][1] == "Factor":
            # For grid cells that have a prediction of 1 standard deviation above the norm multiply the reporting
            enhanced_points = _grid > (mean_std[0] + mean_std[1])
            report_threshold[enhanced_points] = config_dict['BaseRep']*config_dict['RepEn'][2]
        # Where there is a proportional increase w.r.t the PredPol prediction
        elif config_dict['RepEn'][1] == "Proportional":
            # Enhance each grid cell (up to a maximum of 0.8) proportionally to the prediction if it is above
            # mean-std (no enhance) to mean+std (0.8)
            # apply nan filtering where necessary
            _enhance = np.nan_to_num(_grid - (mean_std[0]-mean_std[1]))  # cells with value mean-std now equal 0
            _enhance[_enhance < 0] = 0  # remove negatives
            if mean_std[1] > 0.:
                _enhance = _enhance/(2*mean_std[1])
            _enhance[_enhance > 1] = 1
            report_threshold = report_threshold + ((config_dict['RepEn'][2]-config_dict['BaseRep']) * _enhance)
        elif config_dict['RepEn'][1] is None:
            logging.info("Report enhancement model not given.")
            raise ValueError("Report enhancement model not given.")
        else:
            logging.info("Report enhancement model not found.")
            raise ValueError("Report enhancement model not found.")
    # Generate an random number [0, 1) for each crime and compare to the threshold in the crimes cell to determine if a
    # given crime is recorded. The implicit assumption is made here that every reported crime is also recorded.
    _mask = [np.random.random() < report_threshold[_ix]
             for _ix in np.array(_crime['grid_num'].values).astype(int)]
    logging.debug("RepThresh:\n{}\nMask:\n{}\n".format(report_threshold, _mask))
    logging.info(":GetCrimeReports")
    return np.fliplr(_crime[_mask].values.astype(np.float32)), report_threshold


def plot_crime_profile(_grid, _crime, time_frame, _start_time, plot_name,
                       return_fig=False, fig_ax=None, fig_loc=None):
    """
    :param _grid: array(n,n), Grid passed to get shape/size
    :param _crime: pd.DataFrame, Dataframe of crimes to plot on the heatmap
    :param time_frame: int/float, the time frame to look over for plotting the crimes (positive looks at predictive
    power negative at fitting power)
    :param _start_time: int/float, the time to look forward/backward from
    :param plot_name: string, name of the plot
    :param return_fig: bool, Used to determine if figure is saved or the plot is returned to be used
    :param fig_ax: mpl figure, The figure on which to plot
    :return: optional, figure axis
    """
    # get the grid side
    grid_side = _grid.shape[0]
    # get the crimes between the start and end time
    _crime = _crime[_crime['time'].between(_start_time, _start_time+time_frame)]
    # get the number of crimes in each grid number
    count_series = _crime['grid_num'].value_counts()
    grid_num = count_series.index  # grid indexes
    _count = count_series.values  # grid counts
    x = grid_num % grid_side  # get the x coordinates for the grid
    y = grid_num // grid_side  # get the y coordinates for the grid
    data = pd.DataFrame(data={'x': x, 'y': y, 'z': _count})  # construct a data frame using the x y and count
    data = data.pivot(index='y', columns='x', values='z')  # Piviot the table due to the way the heatmap plots

    fig_ax[1] = sns.heatmap(data, ax=fig_ax[1])
    if return_fig:
        # Return the figure axis for further elements to be added
        return fig_ax
    else:
        # save the figure
        fig_ax[1].set_xlim(0, grid_side)
        fig_ax[1].set_ylim(0, grid_side)
        plt.tight_layout()
        if fig_loc is not None:
            # check if the save directory has be overridden
            plt.savefig(fig_loc + plot_name + ".png")
        else:
            plt.savefig(figure_dir+"crim_theory_eff/"+plot_name+".png")


def plot_predpol(_cond_intn, plot_name, return_fig=False, fig_ax=None, fig_loc=None):
    """
    :param _cond_intn: array(n,n), Values of conditional intensities at the given time-step
    :param plot_name: string, Name of the plot.
    :param return_fig: bool, Used to determine if figure is saved or the plot is returned to be used.
    :param fig_ax: mpl figure, The figure on which to plot.
    :param fig_loc: string, Location to save the figure.
    :return: optional, figure axis
    """

    contours = fig_ax[1].contour(_cond_intn, cmap='GnBu', origin='lower', alpha=0.5)
    fig_ax[0].colorbar(contours, shrink=0.8, extend='both')
    if return_fig:
        return fig_ax
    else:
        fig_ax[1].set_xlim(0, _cond_intn.shape[0])
        fig_ax[1].set_ylim(0, _cond_intn.shape[1])
        plt.xlim(0, _cond_intn.shape[0])
        plt.ylim(0, _cond_intn.shape[0])
        plt.xticks(np.linspace(0, _cond_intn.shape[0], 11), np.linspace(0, _cond_intn.shape[0], 11))
        plt.yticks(np.linspace(0, _cond_intn.shape[0], 11), np.linspace(0, _cond_intn.shape[0], 11))
        plt.tight_layout()
        if fig_loc is not None:
            plt.savefig(fig_loc + plot_name + ".png")
        else:
            plt.savefig(figure_dir+"crim_theory_eff/"+plot_name+".png")


def get_crimes(_grid, number, time_frame, _start_time, model='gaussian', model_dict={'hot_spots': None,
                                                                                     'fixed_spots': None,
                                                                                     'real_mean': None,
                                                                                     'real_std': None}):
    """
    Generate new crimes in the current grid and return them in an data frame, crimes that spill out of the grid are
    returned in numpy arrays.
    :param _grid: array(n,n) grid passed to get shape/size
    :param number: int, number of crimes to seed
    :param time_frame: int/float, time-frame to assign crimes in
    :param _start_time: int/float, beginning of the time frame in which to seed crimes
    :param model: str, the model to use for assigning crimes
    :param model_dict: dictionary with keys that define the crime seeding model
    :return df_crime_hist: pandas data frame storing the crime histories
    :return x_out: array int (M) x coordinates of crimes outside of grid
    :return y_out: array int (M) y coordinates of crimes outside of grid
    :return crime_time_out: array float (M) timestamps of crimes outside grid
    """
    logging.debug("Grid:\n{}\nNumber:\n{}\nTimeFrame:\n{}\nStartTime:\n{}\nModel:\n{}\nModelDict".format(_grid, number,
                                                                                                         time_frame,
                                                                                                         _start_time,
                                                                                                         model,
                                                                                                         model_dict))
    # get the range and side length of the grid
    grid_range = _grid.size
    grid_side = _grid.shape[0]
    # create empty output arrays to store the x and y coordinates of generate crimes
    x_out = np.array([])
    y_out = np.array([])
    # attempt to put the same number (or as close as possible) of crimes in each cell, fail if at least one cannot be
    # assigned to each cell
    if model == 'uniform':
        assert number >= grid_range, "grid is larger than crimes: uniform assignment not possible"
        crimes_per_cell = number // grid_range
        grid_crimes = np.arange(grid_range, dtype=np.int).repeat(crimes_per_cell)
        number = grid_crimes.size
    # for each crime randomly assign it a grid cell
    if model == 'gaussian':
        grid_crimes = np.random.randint(grid_range, size=(number,))
    # check if we are using one of the multivariate norm models
    if (model == 'biased') or (model == 'hot_spot'):
        assert (number//grid_side) >= grid_side, "Not enough crimes to create hot spots"
        # generate hotspots or use those passed to the function
        if model_dict['hot_spots'] is None:
            # if hot_spot_xy is empty
            # pick n (for now n = sqrt grid_range div 2) hot spots assign the crimes in a 2d gaussian around these hot
            # spots.
            hot_spot_xy = np.random.randint(grid_side, size=(10, 2))
            mean_crimes = ((3 * number) // 4) // hot_spot_xy.shape[0]
            hot_spot_xy_itn = np.random.normal(mean_crimes, np.sqrt(mean_crimes), hot_spot_xy.shape[0]).astype(np.int)
        else:
            hot_spot_xy, hot_spot_xy_itn = model_dict['hot_spots']
        # make sure all intensities have >=1 crime
        hot_spot_xy_itn[hot_spot_xy_itn < 1] = 1
        logging.debug("hot_spot_xy\n{}\nhot_spot_xy_itn\n{}\n".format(hot_spot_xy, hot_spot_xy_itn))
        # assign remaining crimes as a background
        background_num = number - np.sum(hot_spot_xy_itn)
        if background_num <= 1:
            background_num = 2
        grid_crimes = np.random.randint(grid_range, size=background_num)

        # for each hotspot position create a 2d gaussian (multivariate normal) around the point
        for _ix in range(len(hot_spot_xy)):
            # get the position and intensity
            x_y = hot_spot_xy[_ix]
            x_y_itn = hot_spot_xy_itn[_ix]
            # get the distribution of crime positions in x and y (round to the nearest integer)
            x, y = np.round(np.random.multivariate_normal(x_y, [[5, 0], [0, 5]], x_y_itn).T).astype(np.int)

            # check the x and y lie within the current grid then create a compined mask for those that are
            _mask_x = np.ma.masked_inside(x, 0, grid_side-1).mask
            _mask_y = np.ma.masked_inside(y, 0, grid_side-1).mask
            _mask = _mask_x & _mask_y

            # get the x and y coordinates for crimes in the grid
            x_g = x[_mask]
            y_g = y[_mask]
            # get the x and y coordinates for crimes out of the grid
            x_out = np.append(x_out, x[~_mask])
            y_out = np.append(y_out, y[~_mask])
            # flatten into grid
            if x_g.size > 0:
                grid_crimes = np.concatenate((grid_crimes, (x_g+(grid_side*y_g))))
        # get the total number of crimes within the grid
        number = grid_crimes.size
    # single cell fixed models
    if model == 'fixed':
        # read the cell location from the model dictionary
        fixed_spot_xy, fixed_spot_xy_per = model_dict['fixed_spots']
        # set the intensity for each fixed cell
        fixed_spot_xy_itn = int((fixed_spot_xy_per/100)*number)
        # get the background number
        background_num = number - fixed_spot_xy_itn*len(fixed_spot_xy)
        logging.debug("fixed_spot_xy_itn\n{}\nbackground_num\n{}\n".format(fixed_spot_xy_itn, background_num))
        # make a random background with the remaining crimes
        if background_num <= 1:
            logging.info("Not enough crimes for background")
            background_num = 2
        grid_crimes = np.random.randint(grid_range, size=background_num)
        logging.debug("grid_crimes:\n{}\n".format(grid_crimes))
        # Check for crimes outside the grid
        for _ix in range(len(fixed_spot_xy)):
            x = np.repeat(fixed_spot_xy[0], fixed_spot_xy_itn)
            y = np.repeat(fixed_spot_xy[1], fixed_spot_xy_itn)

            _mask_x = np.ma.masked_inside(x, 0, grid_side-1).mask
            _mask_y = np.ma.masked_inside(y, 0, grid_side-1).mask
            _mask = _mask_x & _mask_y

            x_g = x[_mask]
            y_g = y[_mask]
            x_out = np.append(x_out, x[~_mask])
            y_out = np.append(y_out, y[~_mask])
            # flatten into grid
            if x_g.size > 0:
                grid_crimes = np.concatenate((grid_crimes, (x_g+(grid_side*y_g))))
        number = grid_crimes.size
    # Construct crimes to mimic the distribution and cadence of real crimes
    if model == 'real_fake':
        assert len(model_dict['real_mean']) == grid_range, "Error: model_dict['real_mean'] is the incorrect " \
                                                           "size: {} != {}".format(len(model_dict['real_mean']),
                                                                                   grid_range)
        # get the mean and standard for this section of the grid
        grid_crimes = np.concatenate([np.full(np.max([number.round().astype(int), 1]), ix)
                                      if (number > np.random.random())
                                      else np.full(0, ix)
                                      for ix, number in
                                      enumerate(np.divide(np.random.normal(model_dict['real_mean'],
                                                                           model_dict['real_std']), 30))
                                      ])
        number = grid_crimes.size

    # Assign times for crimes
    crime_time = np.random.uniform(_start_time, _start_time+time_frame, (number,))
    crime_time_out = np.random.uniform(_start_time, _start_time+time_frame, x_out.size)
    # Turn crimes into a dictionary to return
    df_crime_hist = pd.DataFrame.from_dict({'grid_num': grid_crimes, 'time': crime_time})

    return df_crime_hist, x_out, y_out, crime_time_out


def correlation_function(_grid, _prediction, _reported_crime, _true_crime, time_frame, _start_time, plot_name,
                         hysteresis=False, return_fig=False, fig_axs=None, fig_loc=None):
    """
    Plot the correlation between the crime prediction and the actual crime/ reported crime and the correlation between
    the actual/reported crime. Plot on 3 subplots with linear best fits and legends to determine the strength of the
    correlation.
    :param _grid: array(n,n) grid passed to get shape/size
    :param _prediction: array(N) conditional intensity prediction at _start_time
    :param _reported_crime: pd.DataFrame, dataframe of reported crimes to plot on the heatmap
    :param _true_crime: pd.DataFrame, dataframe of crimes to plot on the heatmap
    :param time_frame: int/float, the time interval to look forward/backward
    :param _start_time: int/float, the time to look forward/backward from
    :param plot_name: string, name of the plot
    :param hysteresis: bool, flag to colour points by their bias at day 0
    :param return_fig: bool, flag to set if the function should return or save a figure
    :param fig_axs: matplotlib figure, axes to add plot elements to
    :param fig_loc: str, a string to override the save location for the figure
    """
    if hysteresis is False:
        hysteresis = np.full_like(_prediction, True, dtype=bool)
    grid_side = _grid.shape[0]
    grid_range = _grid.size
    # filter and get values
    _reported_crime = _reported_crime[_reported_crime['time'].between(_start_time, _start_time + time_frame)]
    _reported_crime_count = _reported_crime['grid_num'].value_counts()
    _reported_crime_grid_num = _reported_crime_count.index.values
    _reported_crime_full = pd.Series(np.zeros((grid_range,)))
    _reported_crime_full[_reported_crime_grid_num] = _reported_crime_count.values
    # normalise values
    _reported_crime_count_norm = (_reported_crime_full/np.sum(_reported_crime_full))

    # filter and get values
    _true_crime = _true_crime[_true_crime['time'].between(_start_time, _start_time + time_frame)]
    _true_crime_count = _true_crime['grid_num'].value_counts()
    _true_crime_grid_num = _true_crime_count.index.values
    _true_crime_full = pd.Series(np.zeros((grid_range,)))
    _true_crime_full[_true_crime_grid_num] = _true_crime_count.values
    # normalise values
    _true_crime_count_norm = (_true_crime_full/np.sum(_true_crime_full))

    # normalise prediction values
    _prediction_norm = (np.divide(_prediction, _prediction.sum()))

    # get axis maximums
    _max = np.max([np.max(_reported_crime_count_norm), np.max(_true_crime_count_norm), np.max(_prediction_norm)])
    _max_b = np.max([np.max(_reported_crime_count_norm), np.max(_true_crime_count_norm)])

    # scatter plots
    fig_axs[1][0, 0].scatter(_reported_crime_count_norm[hysteresis], _prediction_norm[hysteresis], marker='x')
    fig_axs[1][1, 1].scatter(_true_crime_count_norm[hysteresis], _prediction_norm[hysteresis], marker='x')
    fig_axs[1][0, 1].scatter(_true_crime_count_norm[hysteresis], _reported_crime_count_norm[hysteresis], marker='x')

    # Least squares linear fits, try excepted for safety
    x = np.linspace(0, _max, 5)
    try:
        ax1_fit = np.polynomial.polynomial.Polynomial.fit(_reported_crime_count_norm[hysteresis],
                                                          _prediction_norm[hysteresis], 1).convert().coef
        fig_axs[1][0, 0].plot(x, ax1_fit[0] + ax1_fit[1] * x, label="c: {:.4f}, m: {:.2f}".format(ax1_fit[0],
                                                                                                  ax1_fit[1]))
    except LinAlgError as e:
        pass
    try:
        ax2_fit = np.polynomial.polynomial.Polynomial.fit(_true_crime_count_norm[hysteresis],
                                                          _prediction_norm[hysteresis], 1).convert().coef
        fig_axs[1][1, 1].plot(x, ax2_fit[0] + ax2_fit[1] * x, label="c: {:.4f}, m: {:.2f}".format(ax2_fit[0],
                                                                                                  ax2_fit[1]))
    except LinAlgError as e:
        pass
    try:
        ax3_fit = np.polynomial.polynomial.Polynomial.fit(_true_crime_count_norm[hysteresis],
                                                          _reported_crime_count_norm[hysteresis], 1).convert().coef
        fig_axs[1][0, 1].plot(x, ax3_fit[0] + ax3_fit[1] * x, label="c: {:.4f}, m: {:.2f}".format(ax3_fit[0],
                                                                                                  ax3_fit[1]))
    except LinAlgError as e:
        pass
    if return_fig:
        return fig_axs
    else:
        fig_axs[1][0, 0].set_xlim(0, _max)
        fig_axs[1][0, 0].set_ylim(0, _max)
        fig_axs[1][1, 1].set_xlim(0, _max)
        fig_axs[1][1, 1].set_ylim(0, _max)
        fig_axs[1][0, 1].set_xlim(0, _max_b)
        fig_axs[1][0, 1].set_ylim(0, _max_b)
        fig_axs[1][0, 0].set_ylabel("PredPol")
        fig_axs[1][1, 1].set_ylabel("PredPol")
        fig_axs[1][1, 1].set_xlabel("True Crime")
        fig_axs[1][0, 0].set_xlabel("Reported Crime")
        fig_axs[1][0, 1].set_xlabel("True Crime")
        fig_axs[1][0, 1].set_ylabel("Reported")

        fig_axs[1][0, 0].legend(frameon=False)
        fig_axs[1][1, 1].legend(frameon=False)
        fig_axs[1][0, 1].legend(frameon=False)

        fig_axs[0].delaxes(fig_axs[1][1, 0])
        plt.tight_layout()
        if fig_loc is not None:
            plt.savefig(fig_loc + plot_name + ".png")
        else:
            plt.savefig(figure_dir+"crim_theory_eff/"+plot_name+".png")


def kmeans_cluster_plot(_grid, _reported_crime, _true_crime, time_frame, _start_time,
                        plot_name, fig_loc=None, make_plot=True):
    """
    :param _grid: array(n,n) grid passed to get shape/size
    :param _reported_crime: pd.DataFrame, dataframe of reported crimes to plot on the heatmap
    :param _true_crime: pd.DataFrame, dataframe of crimes to plot on the heatmap
    :param time_frame: int/float, the time interval to look forward/backward
    :param _start_time: int/float, the time to look forward/backward from
    :param plot_name: string, name of the plot
    :param fig_loc: string, override plot save location
    :param make_plot: bool, toggle to return or save the figure
    """

    grid_range = _grid.size
    # filter and get values
    _reported_crime = _reported_crime[_reported_crime['time'].between(_start_time, _start_time + time_frame)]
    _reported_crime_count = _reported_crime['grid_num'].value_counts()
    _reported_crime_grid_num = _reported_crime_count.index.values
    _reported_crime_full = pd.Series(np.zeros((grid_range,)))
    _reported_crime_full[_reported_crime_grid_num] = _reported_crime_count.values
    # normalise values
    _reported_crime_count_norm = (_reported_crime_full / np.sum(_reported_crime_full))

    # filter and get values
    _true_crime = _true_crime[_true_crime['time'].between(_start_time, _start_time + time_frame)]
    _true_crime_count = _true_crime['grid_num'].value_counts()
    _true_crime_grid_num = _true_crime_count.index.values
    _true_crime_full = pd.Series(np.zeros((grid_range,)))
    _true_crime_full[_true_crime_grid_num] = _true_crime_count.values
    # normalise values
    _true_crime_count_norm = (_true_crime_full / np.sum(_true_crime_full))

    # stack data
    stacked_data = np.vstack((_true_crime_count_norm, _reported_crime_count_norm)).T

    # pick the number of clusters and make the predctions of clustering
    n_clusters = 2
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    clusterer.fit(stacked_data)
    y_kmeans = clusterer.predict(stacked_data)

    # get the cluster centers
    centers = clusterer.cluster_centers_
    cluster_labels = clusterer.fit_predict(stacked_data)
    # get the silhouette average to predict is there has been
    silhouette_avg = silhouette_score(stacked_data, cluster_labels)

    if make_plot:
        plt.scatter(stacked_data[:, 0], stacked_data[:, 1], c=y_kmeans, s=50, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        plt.title("For n_clusters ="+str(n_clusters)+" silhouette_score is :{:02f}".format(silhouette_avg))

        plt.tight_layout()
        if fig_loc is not None:
            plt.savefig(fig_loc + plot_name + "_" + str(n_clusters) + ".png")
        else:
            plt.savefig(figure_dir + "crim_theory_eff/" + plot_name + "_" + str(n_clusters) + ".png")
    return silhouette_avg
