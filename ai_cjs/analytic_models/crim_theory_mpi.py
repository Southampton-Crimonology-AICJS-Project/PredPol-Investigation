from copy import copy
import logging
from mpi4py import MPI
import numpy as np
import pandas as pd
import sys
from time import time as get_sys_time
from time import strftime

from ai_cjs import cte
from ai_cjs import pp
from ai_cjs.config import local_scratch, config_dict


def bounds_check(_mask_tup):
    """
    checks if the intended send/receive location exits and sends back a list of parameters to be used for the send
    :_mask_tup: tuple (np.mask, (int, int)) mask for array of send data, tuple for direction of send
    :return: list [np.mask, int, bool, (int, int)] mask for array of send data, rank to be sent to, bool True if send
    location exists, tuple for direction of send
    """

    logging.debug("Bounds Check:")
    logging.debug('MaskTup:\n{}\n'.format(_mask_tup))

    # switches to False if the send location does not exist
    ret_bool = True
    # split the mask (to be past through) and the (x,y) vector
    passthrough, tup = _mask_tup
    # get row number by doing a floor division
    row_num = rank // row

    # Check if xp is valid
    if tup[0] == 1:
        ret_bool = ret_bool & \
                   (((rank + 1) % row) != 0)
    # Check if xn is valid
    if tup[0] == -1:
        ret_bool = ret_bool & \
                   ((rank + 1 - (row * row_num)) != 1)
    # Check if yn is valid
    if tup[1] == -1:
        ret_bool = ret_bool & \
                   (row_num != 0)
    # Check if yp is valid
    if tup[1] == 1:
        ret_bool = ret_bool & \
                   ((row_num + 1) != row)
    # create an output list
    out = [passthrough, int((rank + tup[0]) + (tup[1] * row)), ret_bool, tup]
    logging.debug('Out:\n{}\n'.format(out))
    logging.debug(":Bounds Check")
    return out


def send_recv(_x_out, _y_out, _crime_time_out, today="init"):
    """
    function to send and receive crimes from neighbouring grids
    :param _x_out: x coordinates of crimes leaving the cell
    :param _y_out: y coordinates of crimes leaving the cell
    :param _crime_time_out: timestamps of crimes leaving the cell
    :param today: the 'day'
    :return: dataframe of gridnumbers and times of crimes entering the cell
    """

    logging.debug("SendReceive:")
    logging.debug("_x_out:\n{}\n _y_out:\n{}\n _crime_time_out:\n{}\n today:\n{}\n".format(_x_out, _y_out,
                                                                                           _crime_time_out, today))

    # get the orthogonal masks
    x_p_mask = np.ma.masked_where(_x_out > sub_grid_n - 1, _x_out).mask
    x_n_mask = np.ma.masked_where(_x_out < 0, _x_out).mask
    y_p_mask = np.ma.masked_where(_y_out > sub_grid_n - 1, _y_out).mask
    y_n_mask = np.ma.masked_where(_y_out < 0, _y_out).mask

    # get the rank_masks that combine orthogonal masks
    # xp: x positive
    # xn: x negative
    # yp: y positive
    # yn: y negative
    xp = x_p_mask & ~(y_p_mask | y_n_mask)
    diag_xpyn = x_p_mask & y_n_mask
    yn = y_n_mask & ~(x_p_mask | x_n_mask)
    diag_xnyn = x_n_mask & y_n_mask
    xn = x_n_mask & ~(y_p_mask | y_n_mask)
    diag_xnyp = x_n_mask & y_p_mask
    yp = y_p_mask & ~(x_p_mask | x_n_mask)
    diag_xpyp = x_p_mask & y_p_mask

    # make sure the masks sum to the number of cells
    assert (np.sum(xp) + np.sum(yp) + np.sum(xn) + np.sum(yn) +
            np.sum(diag_xpyn) + np.sum(diag_xnyn) + np.sum(diag_xnyp) + np.sum(diag_xpyp)) ==\
        x_out.size, "Assertion Fail: masks do not sum."

    # create the list of bounds to check consisting of mask direction pairs
    bounds = [(xp, (1, 0)),
              (diag_xpyn, (1, -1)),
              (yn, (0, -1)),
              (diag_xnyn, (-1, -1)),
              (xn, (-1, 0)),
              (diag_xnyp, (-1, 1)),
              (yp, (0, 1)),
              (diag_xpyp, (1, 1))]
    # pass the bounds to the check and get the send information
    rank_list = [bounds_check(mask_tup) for mask_tup in bounds]
    # get the data to send out
    sends = [np.concatenate((_x_out[tup[0]],
                            _y_out[tup[0]],
                            _crime_time_out[tup[0]]))
             if tup[2] else None for tup in rank_list]

    # Assert the number of cells to send to are 3 5 or 8 which are the only three valid numbers in the geometry
    assert np.any(sum([1 if i is not None else 0 for i in sends]) == np.array([3, 5, 8])), "Sending to incorrect " \
                                                                                           "number of ranks. " \
                                                                                           "rank: {}".format(rank)
    # push day message to stdout
    logging.info("I am :{} on day: {} sending,".format(rank, today))
    # send the crimes that spill out of the grid
    send_list = []
    for _ix, tup in enumerate(rank_list):
        if tup[2]:
            send_list.append(comm.Isend(sends[_ix].astype(np.float), dest=tup[1], tag=tup[1]))
            logging.debug("Rank {}: SendList for {}\n{}\n".format(rank, tup[1], sends[_ix].astype(np.float)))
    # Assert the number of cells to send to are 3 5 or 8 which are the only three valid numbers in the geometry
    assert np.any(len(send_list) == np.array([3, 5, 8])), "\nSent to incorrect number of ranks. rank: {}".format(rank)
    comm.Barrier()
    # set up data-frame to receive from other grids
    _crimes_from_other_grids = pd.DataFrame(columns=['grid_num', 'time'])
    # receive crimes that have come from other grids
    while sum(item[2] for item in rank_list) > 0:
        recv_status = MPI.Status()  # get MPI status object
        comm.Probe(source=MPI.ANY_SOURCE, tag=rank, status=recv_status)  # probe for a send
        _count = recv_status.Get_count()  # get the size of the object sent: Note python float is MPI double
        data = np.empty(_count, dtype=np.float)  # create an array large enough to store the
        comm.Recv(data, status=recv_status)  # get the sent data
        # get rid of the flag that waits for data
        pop_ix = [x[1] if x[2] else None for x in rank_list].index(recv_status.Get_source())
        # if the data had a size
        if _count > 0:
            data_cut = _count // 3  # create an index to separate the 3 parts (x,y,time) of the data set
            # x values data[:data_cut], y values data[data_cut:data_cut*2], times data[data_cut*2:]
            # get the sign of the data in x and y
            x_pn, y_pn = rank_list[pop_ix][3]
            # validate the data
            assert np.all(-np.sign(x_pn) == np.sign(data[:data_cut])) or np.sign(x_pn) == 0, \
                "\nIncorrect x values. Rank: {}, Source: {}, xud: {}\n xval: {}".format(rank, recv_status.Get_source(),
                                                                                        -np.sign(x_pn), data[:data_cut])
            assert np.all(-np.sign(y_pn) == np.sign(data[data_cut:2*data_cut])) or np.sign(y_pn) == 0, \
                "\nIncorrect y values. Rank: {}, Source: {}, yud: {}\n yval: {}".format(rank, recv_status.Get_source(),
                                                                                        -np.sign(y_pn),
                                                                                        data[data_cut:2*data_cut])
            # transform from the 'out of grid num' from its source to its 'current grid num' in its destination
            _grid_nums = ((data[:data_cut] % sub_grid_n) + sub_grid_n * (data[data_cut:data_cut*2] % sub_grid_n))
            # cut the times
            _times = data[data_cut*2:]
            # append the grid number and the time to a dataframe
            _crimes_from_other_grids = _crimes_from_other_grids.append(pd.DataFrame.from_dict(
                {'grid_num': _grid_nums, 'time': _times}),
                ignore_index=True)
        # pop the received data
        if pop_ix is not None:
            rank_list.pop(pop_ix)
    # report that rank is done and waiting for completion
    logging.info("waiting: {} rank:{}.".format(today, rank))
    # sync threads
    comm.Barrier()
    # make sure MPI has completed the sends
    MPI.Request.Waitall(send_list)
    # sync threads
    comm.Barrier()
    # print to stdout that all have been received
    logging.info("All recv day: {} rank:{}.".format(today, rank))
    # return the data frame with the crimes from outside the grid
    logging.debug(":SendReceive")
    return _crimes_from_other_grids


def get_global_mean_std(local_count):
    """
    function to calculate the global mean and standard deviation
    :param local_count: pandas series of crimes per grid cell in given timeframe
    :return: the mean and standard deviation globally
    """

    logging.debug('GetGlobalMeanStd:')
    logging.debug('LocalCount:\n{}\n'.format(local_count))

    # nan masking
    nan_mask = np.isfinite(local_count)  # only use finite numbers in the calculation
    non_nan_elements = comm.allreduce(np.sum(nan_mask.astype(int)), op=MPI.SUM)  # cast to int and sum to get the
    logging.debug('NonNan:\n{}\n'.format(non_nan_elements))
    # number of element being used in the mean and std

    # mean
    total = np.sum(local_count[nan_mask])  # get the local (non nan) total
    total = comm.allreduce(total, op=MPI.SUM)  # get the global total
    _mean = total/non_nan_elements  # get the global mean

    # std
    sq_diff = np.sum(np.power(local_count[nan_mask] - _mean, 2))  # calculate local (non nan) square difference
    sq_diff = comm.allreduce(sq_diff, op=MPI.SUM)  # get the global square difference
    _std = np.sqrt(sq_diff/non_nan_elements)  # get the global standard deviation

    logging.debug(':GetGlobalMeanStd')
    return _mean, _std


def parse_crime_model(_model, _crime_history, _cond_intn, _crime_model_dict=None):

    logging.debug("ParseCrimeModel:")
    logging.debug("Model:\n{}\nCrimeHistory\n{}\nCondIntn\n{}\n".format(_model, _crime_history, _cond_intn))
    if _crime_model_dict is None:
        _crime_model_dict = {
            'hot_spots': None,
            'fixed_spots': None,
            'real_mean': None,
            'real_std': None
        }
    if _model == 'hot_spot' and (_crime_history is not None):
        # get crime from last x days, then calculate the global mean and standard deviation
        prev_crime = _crime_history[_crime_history['time'] > _time - 5]
        crime_counts = prev_crime['grid_num'].value_counts()
        mean, std = get_global_mean_std(crime_counts.values)

        # For any cell that exceeds the mean and standard deviation create a hot spot with intensity equal to its
        # previous crime +/- the standard deviation
        grid_nums = crime_counts[crime_counts.values > (mean + std)].index.values.astype(np.int)
        hot_spots_xy = [[_num % sub_grid_n, _num // sub_grid_n] for _num in grid_nums]
        intensities = np.random.normal(crime_counts[crime_counts.values > (mean + std)].values, std).astype(np.int)
        if len(hot_spots_xy) > 0:
            _crime_model_dict['hot_spots'] = (hot_spots_xy, intensities)
        logging.debug("hot_spots_xy:\n{}\nintensities:\n{}\n".format(hot_spots_xy, intensities))
    # return dictionary for 'get_crimes'
    if _model == 'fixed':
        percent = config_dict['FixedPer']
        if percent > 49:
            percent //= 2
        # check if the fixed model have been run
        if _crime_model_dict['fixed_spots'] is None:
            # for first run create two fixed spots per core
            _crime_model_dict['fixed_spots'] = ([[np.random.randint(0, sub_grid_n), np.random.randint(0, sub_grid_n)],
                                                 [np.random.randint(0, sub_grid_n), np.random.randint(0, sub_grid_n)]],
                                                percent)
            # make sure the fixed spots do not overlap
            while _crime_model_dict['fixed_spots'][0] == _crime_model_dict['fixed_spots'][1]:
                _crime_model_dict['fixed_spots'][1] = [np.random.randint(0, sub_grid_n),
                                                       np.random.randint(0, sub_grid_n)]
        logging.debug('fixed_spots:\n{}\nPercent:\n{}\n'.format(_crime_model_dict['fixed_spots'][0],
                                                                _crime_model_dict['fixed_spots'][1]))
    # get the distribution of means and standard deviations for the 'real rake' model
    if _model == 'real_fake':
        _crime_model_dict['real_mean'] = config_dict['RealFake'][0][sub_grid_n**2 * rank: sub_grid_n**2 * (rank + 1)]
        _crime_model_dict['real_std'] = config_dict['RealFake'][1][sub_grid_n**2 * rank: sub_grid_n**2 * (rank + 1)]
        logging.debug("real_mean:\n{}\nreal_std:\n{}z\n".format(_crime_model_dict['real_mean'],
                                                                _crime_model_dict['real_std']))
    logging.debug(":ParseCrimeModel")
    return _crime_model_dict


def aggregate_and_save(ci, ch, cr, rt, final=False):
    # Reindex and save outputs =========================================================================================

    # create a copy of the original data so not to reindex the working values
    conditional_intensity = copy(ci)
    _crime_history = copy(ch)
    _crime_reports = copy(cr)
    _rep_thresh = copy(rt)

    # remove init conditional intensity row
    conditional_intensity = conditional_intensity[1:]

    # re-index distribute concatenate and save the data in total_crime_history
    x_grid = np.array(_crime_history['grid_num'].values) % sub_grid_n
    y_grid = np.array(_crime_history['grid_num'].values) // sub_grid_n
    x_rank, y_rank = rank % row, rank // row
    grid_glob = (n_max * row * y_rank) + (sub_grid_n * x_rank) + (y_grid * sub_grid_n * row) + x_grid
    _crime_history['grid_num'] = grid_glob
    total_crime_history = pd.concat(comm.allgather(_crime_history))
    # re-index distribute concatenate and save the data in total_crime_reports
    x_grid = _crime_reports[:, 1] % sub_grid_n
    y_grid = _crime_reports[:, 1] // sub_grid_n
    x_rank, y_rank = rank % row, rank // row
    grid_glob = (n_max * row * y_rank) + (sub_grid_n * x_rank) + (y_grid * sub_grid_n * row) + x_grid
    _crime_reports[:, 1] = grid_glob
    total_crime_reports = np.vstack(comm.allgather(_crime_reports))

    # distribute then reindex the conditional intensities
    total_cond_intn = np.hstack(comm.allgather(conditional_intensity))
    index_arr = np.concatenate([
                    [
                        (n_max * row * (jx // row)) +
                        (sub_grid_n * (jx % row)) +
                        ((nx // sub_grid_n) * sub_grid_n * row) +
                        nx % sub_grid_n
                        for nx in range(n_max)]
                    for jx in range(size)])
    total_cond_intn_reidx = np.zeros_like(total_cond_intn)
    for ix in range(len(total_cond_intn)):
        total_cond_intn_reidx[ix] = total_cond_intn[ix][index_arr.argsort()]

    # distribute then reindex the rep_thresh
    total_rep_thresh = np.hstack(comm.allgather(_rep_thresh))
    index_arr = np.concatenate([
                    [
                        (n_max * row * (jx // row)) +
                        (sub_grid_n * (jx % row)) +
                        ((nx // sub_grid_n) * sub_grid_n * row) +
                        nx % sub_grid_n
                        for nx in range(n_max)]
                    for jx in range(size)])
    total_rep_thresh_reidx = np.zeros_like(total_rep_thresh)
    for ix in range(len(total_rep_thresh)):
        total_rep_thresh_reidx[ix] = total_rep_thresh[ix][index_arr.argsort()]

    # if the save is a checkpoint append it with date and time
    if final:
        tag = ""
    else:
        tag = strftime("_%Y-%d%H-%M-%S")

    # Save the outputs as numpy binary files
    if rank == 0:
        np.save(local_scratch + 'mpi_out/ch{}.npy'.format(tag), total_crime_history.values)
    elif rank == 1:
        np.save(local_scratch + 'mpi_out/cr{}.npy'.format(tag), total_crime_reports)
    elif rank == 2:
        np.save(local_scratch + 'mpi_out/ci{}.npy'.format(tag), total_cond_intn_reidx)
    elif rank == 3:
        np.save(local_scratch + 'mpi_out/rt{}.npy'.format(tag), total_rep_thresh_reidx)
    # ==================================================================================================================


# run 'mpiexec -n $CORES python ai_cjs/analytic_models/crim_theory_mpi.py'
if __name__ == '__main__':
    run_start_time = get_sys_time()
    # Set up global parameters and initiate MPI=========================================================================
    global line_end
    line_end = "\n"
    conv_otp_end = "\n"
    conv_otp_freq = 100

    # Set up MPI
    global comm, size, rank, row
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    assert np.sqrt(size) % 1 == 0, "Grid ({}) not square: number of cores/grids must be a square number".format(size)
    row = int(np.sqrt(size))
    print("I am rank: ", rank, " of ", size)
    sys.stdout.flush()

    # time conversions and definitions
    month = 31
    day = 1
    # ==================================================================================================================

    # Get command line inputs===========================================================================================
    global sub_grid_n, n, u_grid, n_max
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
    #   #--RepEn: reporting_enhance
    #   #--EnLvl: Level of enhancement
    #   #--FixPer percentage of crime to be seeded in the fixed locations

    # grid_sizes
    sub_grid_n = config_dict['SubGridN']

    # Convergence limit/threshold for predpol algorithm
    conv_threshold = config_dict['ConvThresh']  # get to the point where nothing changes below this threshold
    conv_lim = config_dict['ConvLim']  # break after this many steps


    # get number of days to run
    days_to_run = config_dict['Days']
    # get sizes of arrays
    n = sub_grid_n * row
    u_grid = np.zeros((sub_grid_n, sub_grid_n))
    n_max = u_grid.size
    # create working/output arrays
    cond_intn = np.zeros(n_max)
    rep_thresh = np.zeros(n_max)
    # ==================================================================================================================

    # create the initial crimes
    # =========================================================================================
    # for the initlisation step get the crime history and crime reports
    crime_model_dict = parse_crime_model(config_dict['InitModel'], None, cond_intn)
    # try except to catch and send an mpi abort after flushing the error to log
    try:
        crime_history, x_out, y_out, crime_time_out = cte.get_crimes(u_grid, config_dict['InitCrime'],
                                                                     config_dict['InitDays'], 0,
                                                                     model=config_dict['InitModel'],
                                                                     model_dict=crime_model_dict)
        crimes_from_other_grids = send_recv(x_out, y_out, crime_time_out)
        crime_history = crime_history.append(crimes_from_other_grids, ignore_index=True)
        crime_reports, rep_thresh = cte.get_crime_reports(cond_intn, crime_history, mean_std=(np.inf, np.inf))
        sys.stdout.flush()
    except Exception as e:
        print("Initialisation failed with: {}\nSee log for traceback".format(e))
        logging.info("Initialisation failed with: {}".format(e))
        logging.exception(e)
        sys.stdout.flush()
        comm.Abort()

    # std_mean set to inf to prevent report enhancement in the initialisation step
    # ==================================================================================================================

    # Start the model ==================================================================================================
    if rank == 0:
        logging.info("starting time loop")
        sys.stdout.flush()
    for _time in np.arange(config_dict['InitDays'] + 1, config_dict['InitDays'] + 1 + days_to_run, day):
        # initialise PredPol parameters

        # d_param is jit dict for passing into PP functions using jit
        # _w, _theta, _u are typed lists (for compatibility with d_param)
        d_param = pp.make_dict(_n=n_max)
        _w = np.array([0.5]).astype(np.float32)
        _theta = np.array([0.5]).astype(np.float32)
        _u = np.full_like(d_param['u'], 0.5)

        # Converge w theta phi =========================================================================================
        count = 0
        exit_cond = True
        while exit_cond:
            # get the numerator and denominator then reduce across all ranks
            try:
                num, dom_t, dom_w = pp.get_w_theta_split(d_param, crime_reports, _time, day, n_max)
            except Exception as e:
                print("w, theta failed with:", e)
                sys.stdout.flush()
                comm.Abort()
            # collect the PredPol parameters from the whole grid
            num = comm.allreduce(num, op=MPI.SUM)
            dom_t = comm.allreduce(dom_t, op=MPI.SUM)
            dom_w = comm.allreduce(dom_w, op=MPI.SUM)
            if (num != 0.0) and (dom_w != 0.0):
                _w[0] = np.divide(num, dom_w)
            if (num != 0.0) and (dom_t != 0.0):
                _theta[0] = np.divide(num, dom_t)
            # get the u for each cell
            try:
                _u = pp.get_u(d_param, crime_reports, _time, n_max, t=day)
            except Exception as e:
                print("u failed with:", e)
                sys.stdout.flush()
                comm.Abort()
            # Update the dictionary
            d_param['w'][0], d_param['theta'][0], d_param['u'] = _w[0], _theta[0], _u
            count += 1

            if count > conv_lim:
                logging.info("rank {} hit convergence iteration limit on day: {} \n Aborting".format(rank, _time))
                sys.stdout.flush()
                comm.Abort()

            # Exit condition is that the difference
            # is less that 100*conv_threshold percent of the parameter value
            exit_cond = (np.abs(d_param['w'][0] - _w[0]) > conv_threshold * _w[0]) \
                or (np.abs(d_param['theta'][0] - _theta[0]) > conv_threshold * _theta[0]) \
                or (np.abs(d_param['u'] - _u) > conv_threshold * _u).any()
            # Check exit condition is met on all ranks
            exit_cond = any(comm.allgather(exit_cond))
            # If the exit condition is met (i.e. exit_cond == False) wait on all ranks
            if not exit_cond:
                comm.Barrier()

        if rank == 0:
            logging.info("converged on day: {}".format(_time))
            sys.stdout.flush()
        # ==============================================================================================================

        # Calculate the conditional intensities i.e. make a prediction =================================================
        _dict = pp.make_dict(_n=n_max)
        _dict['u'] = u_grid.flatten().astype(np.float32)
        _dict['theta'][0] = _theta[0]
        _dict['w'][0] = _w[0]
        if len(crime_reports) > 0:
            try:
                cond_intn = np.vstack((cond_intn,
                                       [pp.cond_intensity(_dict, crime_reports, i, crime_reports[:, 1] == i, _time)
                                        for i in range(n_max)]
                                       ))
            except Exception as e:
                print("Conditional intensity prediction failed with: {}\nSee log for traceback".format(e))
                logging.info("Conditional intensity prediction failed with: {}".format(e))
                logging.exception(e)
                sys.stdout.flush()
                comm.Abort()
        # ==============================================================================================================

        # Get new crimes according to modelling ========================================================================
        try:
            crime_model_dict = parse_crime_model(config_dict['DailyModel'], crime_history, cond_intn[-1],
                                                 _crime_model_dict=crime_model_dict)
        except Exception as e:
            print("Parsing crime model failed with: {}\nSee log for traceback".format(e))
            logging.info("Parsing crime model failed with: {}".format(e))
            logging.exception(e)
            sys.stdout.flush()
            comm.Abort()
        try:
            # assign crimes distributed statically over the grid within the time-step
            new_crime, x_out, y_out, crime_time_out = cte.get_crimes(np.zeros((sub_grid_n, sub_grid_n)),
                                                                     config_dict['DailyCrime'], day, _time,
                                                                     model=config_dict['DailyModel'],
                                                                     model_dict=crime_model_dict)
        except Exception as e:
            print("Assigning new crimes failed with: {}\nSee log for traceback".format(e))
            logging.info("Assigning new crimes failed with: {}".format(e))
            logging.exception(e)
            sys.stdout.flush()
            comm.Abort()
        # get crimes from the other ranks
        try:
            crimes_from_other_grids = send_recv(x_out, y_out, crime_time_out, today=_time)
            new_crime = new_crime.append(crimes_from_other_grids, ignore_index=True)
            crime_history = crime_history.append(new_crime, ignore_index=True)
        except Exception as e:
            print("Getting new crimes failed with: {}\nSee log for traceback".format(e))
            logging.info("Getting new crimes failed with: {}".format(e))
            logging.exception(e)
            sys.stdout.flush()
            comm.Abort()
        try:
            # calculate crime reports from background+prediction
            tmp_crime_reports, tmp_rep_thresh = cte.get_crime_reports(cond_intn[-1], new_crime,
                                                                      mean_std=get_global_mean_std(cond_intn[-1]))
            sys.stdout.flush()
            rep_thresh = np.vstack((rep_thresh, tmp_rep_thresh))
            crime_reports = np.concatenate((crime_reports, tmp_crime_reports))
        except Exception as e:
            print("Rank: {} reporting crimes failed with: {}\nSee log for traceback".format(rank, e))
            logging.info("Rank: {} reporting crimes failed with: {}".format(rank, e))
            logging.exception(e)
            sys.stdout.flush()
            comm.Abort()
        # ==============================================================================================================

        # Confirm day time-step has finished and wait for all ranks ====================================================
        comm.Barrier()
        if (_time % 10 == 0) or (((get_sys_time() - run_start_time)/(60*60)) > 58):
            aggregate_and_save(cond_intn, crime_history, crime_reports, rep_thresh)
        if rank == 0:
            logging.info("finished day: {}".format(_time))
            sys.stdout.flush()
        # ==============================================================================================================
    aggregate_and_save(cond_intn, crime_history, crime_reports, rep_thresh, final=True)
    # ==================================================================================================================


