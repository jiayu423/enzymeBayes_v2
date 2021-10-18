import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
import scipy


class enzymeBayes:
    def __init__(self, dirs_, loadFromCvs=True, tot_x=None, tot_y=None, 
                    tot_t=None, tot_mom=None, tot_track_info=None, tot_track_id=None, 
                        tot_msd=None, tot_msd_id=None, BI=None, jump=0, ml=5, fts=1, ptm=1):
        
        if loadFromCvs: 
            tot_x, tot_y, tot_t = [], [], []
            tot_mom, tot_track_info, tot_track_id = [], [], []
            tot_msd, tot_msd_id, BI = [], [], []
            for dir_ in dirs_:
                x, y, t, track_info, track_id, mom = self.loadRawMinData(dir_, min_length=ml, frameToSecond=fts,
                                                                         pixelToMeter=ptm)

                tot_x.append(x)
                tot_y.append(y)
                tot_t.append(t)
                tot_mom.append(mom)
                tot_track_info.append(track_info)
                tot_track_id.append(track_id)
                tot_msd.append([])
                tot_msd_id.append(np.zeros((len(track_info), )))
                BI.append(np.ones((len(track_info), ))*jump)
                
        self.tot_x = tot_x
        self.tot_y = tot_y
        self.tot_t = tot_t

        self.tot_mom = tot_mom
        self.tot_track_info = tot_track_info
        self.tot_track_id = tot_track_id
        self.n_conc = len(tot_x)

        self.tot_msd = tot_msd
        self.tot_msd_id = tot_msd_id

        self.BI = BI

    @staticmethod
    def loadMin(df, minLength, frameToSecond, pixelToMeter, file_number):
        iMax = int(df["Trajectory"].iloc[-1])
        x_lis, y_lis, t_lis = [], [], []
        traj_track = []
        traj_id = []
        mom_lis = []

        for i in range(1, iMax + 1):
            idx = df["Trajectory"] == float(i)
            if np.sum(idx) >= minLength:

                temp_mom_lis = np.zeros((np.sum(idx), 5))

                x_lis.append(df[idx]["x"].to_numpy() * pixelToMeter)
                y_lis.append(df[idx]["y"].to_numpy() * pixelToMeter)
                t_lis.append(df[idx]["Frame"].to_numpy() * frameToSecond)
                traj_track.append(np.sum(idx))
                traj_id.append([file_number, i])

                for k in range(5):
                    m = "m" + str(k)
                    temp_mom_lis[:, k] = df[idx][m].to_numpy()

                mom_lis.append(temp_mom_lis)

        return [x_lis, y_lis, t_lis, traj_track, traj_id, mom_lis]

    @staticmethod
    def combineData(lis):
        # obtain the first traj (from the first item of the list)
        total_var = lis[0][0].reshape(len(lis[0][0]), 1)

        for i in range(len(lis)):

            # for the first item of the list,
            # the start point for the following loop will 1
            if i == 0:
                start = 1
            else:
                start = 0

            # loop through the remaining traj and stack the data
            for j in range(start, len(lis[i])):
                temp_var = lis[i][j].reshape(len(lis[i][j]), 1)
                total_var = np.vstack((total_var, temp_var))

        return total_var.reshape(len(total_var), )

    def loadRawMinData(self,
                       dir_, min_length, frameToSecond=1, pixelToMeter=1
                       ):
        entries = Path(dir_)

        # obtain total number of trajectory files
        file_count = 0
        for entry in sorted(entries.iterdir(), reverse=False):
            if entry.name[0] == ".":
                continue
            file_count += 1

        # initialize empty list to hold data from each csv file
        x_tot_lis, y_tot_lis, t_tot_lis, mom_tot_lis = [], [], [], []

        # initialize empty list to hold length of each trajectory
        track_tot = []

        # initialize empty list to hold trajectory ID
        track_id = []

        # init files name index
        file_index = 0

        # loop through folder to obatin data
        for entry in sorted(entries.iterdir(), reverse=False):
            if entry.name[0] == ".":
                continue

            filename = dir_ + entry.name
            # used to locate the file numbering
            if file_index == 0:
                for i, s in enumerate(entry.name):
                    if s == "-":
                        file_index = i + 1
                        break
            if entry.name[file_index + 3] == "-":
                file_number = entry.name[file_index: file_index + 5]
            else:
                file_number = entry.name[file_index: file_index + 7]
            df = pd.read_csv(filename)
            temp = self.loadMin(
                df, min_length, frameToSecond, pixelToMeter, file_number
            )

            if len(temp[0]) != 0:
                x_tot_lis.append(temp[0])
                y_tot_lis.append(temp[1])
                t_tot_lis.append(temp[2])
                track_tot.append(temp[3])
                track_id.append(temp[4])
                mom_tot_lis.append(temp[5])

        # stack all data point into a 1d array
        x_tot_arr = self.combineData(x_tot_lis)
        y_tot_arr = self.combineData(y_tot_lis)
        t_tot_arr = self.combineData(t_tot_lis)

        track_tot_ = []
        track_id_ = []
        track_mom_ = []
        for i in range(len(track_tot)):
            for j in range(len(track_tot[i])):
                track_tot_.append(track_tot[i][j])
                track_id_.append(track_id[i][j])
                track_mom_.append(mom_tot_lis[i][j])

        del x_tot_lis, y_tot_lis, t_tot_lis, track_tot, track_id, mom_tot_lis

        print(
            "%d files; %d trajectories (length >= %d); Total %d data points"
            % (
                int(file_count),
                len(track_tot_),
                min_length,
                int(x_tot_arr.shape[0]),
            )
        )

        return x_tot_arr, y_tot_arr, t_tot_arr, track_tot_, track_id_, track_mom_

    # use track_info and full data set to obtain a specific trajectory
    def loadSelectTraj(self, conc, index, isMom=False):

        # get the correct concentration
        x, y, t = self.tot_x[conc], self.tot_y[conc], self.tot_t[conc]
        track_info = self.tot_track_info[conc]


        # get the start position of desired trajectory
        start = 0
        i = -1
        for i, length in enumerate((self.tot_track_info)[conc][:index]):
            start += length

        if isMom: 
            mom = self.tot_mom[conc]
            return ( 
            x[start: start + (track_info[i + 1])],
            y[start: start + (track_info[i + 1])],
            t[start: start + (track_info[i + 1])],
            mom[index]
            
           )
            
        else: 
            return (
            x[start: start + (track_info[i + 1])],
            y[start: start + (track_info[i + 1])],
            t[start: start + (track_info[i + 1])],
           )

    @staticmethod
    def breakup(sx, sy, st, mom, longjump):

        new_tracks = []
        sdt = np.diff(st)
        ind_ = np.where(sdt >= longjump)[0]

        if len(ind_) == 0:
            return [[sx, sy, st, mom]]

        for i in range(len(ind_)):
            temp_x = sx[:ind_[i] + 1]
            sx = sx[ind_[i] + 1:]

            temp_y = sy[:ind_[i] + 1]
            sy = sy[ind_[i] + 1:]

            temp_t = st[:ind_[i] + 1] - st[:ind_[i] + 1][0]
            st = st[ind_[i] + 1:]

            temp_mom = mom[:ind_[i] + 1]
            mom = mom[ind_[i] + 1:]

            ind_ = ind_ - ind_[i] - 1

            new_tracks.append([temp_x, temp_y, temp_t, temp_mom])

        st = st - st[0]
        new_tracks.append([sx, sy, st, mom])

        return new_tracks

    def ignoreLongJump(self):

        tot_x, tot_y, tot_t, tot_track_info, tot_mom = [], [], [], [], []
        tot_msd_id, tot_BI = [], []

        for i in range(self.n_conc):
            track_info = self.tot_track_info[i]
            new_x, new_y, new_t, new_track_info, new_mom = [], [], [], [], []
            for j in range(len(track_info)):
                sx, sy, st, mom = self.loadSelectTraj(i, j, isMom=True)

                if self.BI[i][j] == 0: 
                    new_x.append(sx); new_y.append(sy); new_t.append(st); new_track_info.append(len(sx)); new_mom.append(mom)
                    continue

                new_tracks = self.breakup(sx, sy, st, mom, self.BI[i][j])
                for k in range(len(new_tracks)):
                    new_x.append(new_tracks[k][0])
                    new_y.append(new_tracks[k][1])
                    new_t.append(new_tracks[k][2])
                    new_track_info.append(len(new_tracks[k][0]))
                    new_mom.append(new_tracks[k][3])
            new_x = np.concatenate(new_x)
            new_y = np.concatenate(new_y)
            new_t = np.concatenate(new_t)
            tot_x.append(new_x)
            tot_y.append(new_y)
            tot_t.append(new_t)
            tot_track_info.append(new_track_info)
            tot_mom.append(new_mom)
            tot_msd_id.append(np.zeros((len(new_track_info), )))
            tot_BI.append(np.zeros((len(new_track_info), )))

        self.tot_x, self.tot_y, self.tot_t = tot_x, tot_y, tot_t
        self.tot_track_info, self.tot_mom = tot_track_info, tot_mom
        self.tot_msd_id, self.BI = tot_msd_id, tot_BI

    @staticmethod
    def MAP_bm(sdx, sdy, sdt, a=1, b=1):

        # not really map, but better
        alpha = len(sdx) + a
        beta = np.sum((sdx ** 2 + sdy ** 2) / 4 / sdt) + b

        return beta / (alpha - 1)

    @staticmethod
    def dx_lag_pos(sdt, lag):
        auto1, auto2 = [], []

        for i in range(len(sdt) + 1):
            temp_dt = 0
            for j in range(len(sdt[i:])):
                temp_dt += sdt[i + j]
                if temp_dt == lag:
                    auto1.append(i)
                    auto2.append(i + j + 1)
                    break
                elif temp_dt > lag:
                    break
        return auto1, auto2

    @staticmethod
    def calautodx_pos(sdt):
        auto_1, auto_2 = [], []
        for i in range(len(sdt) - 1):
            if sdt[i] == 1 and sdt[i + 1] == 1:
                auto_1.append(i)
                auto_2.append(i + 1)
        return auto_1, auto_2

    @staticmethod
    def tq_meandx(x_, *argv):
        return np.mean(np.diff(x_, axis=0), axis=0)

    @staticmethod
    def tq_stdx(x_, *argv):
        return np.std(x_, axis=0)

    def tq_maxabsdx(self, x_, *argv):
        if x_.ndim == 1:
            x_ = x_.reshape((len(x_), 1))

        sdt, lag = argv[0][0], argv[0][1]
        pos1, pos2 = self.dx_lag_pos(sdt, lag)

        maxdx = np.max(np.abs(x_[pos2, :] - x_[pos1, :]), axis=0)
        return maxdx
    
    def tq_maxdis(self, x_, y_, *argv):
        
        sdt, lag = argv[0][0], argv[0][1]
        
        if x_.ndim == 1:
            x_ = x_.reshape((len(x_), 1))
            y_ = y_.reshape((len(y_), 1))
            
        pos1, pos2 = self.dx_lag_pos(sdt, lag)    
        maxdis = np.max(np.sqrt((x_[pos2, :] - x_[pos1, :])**2 + (y_[pos2, :] - y_[pos1, :])**2), axis=0)
        return maxdis
    
    def tq_msd(self, x_, y_, *argv):

        sdt, lag = argv[0][0], argv[0][1]
        
        if x_.ndim == 1:
            x_ = x_.reshape((len(x_), 1))
            y_ = y_.reshape((len(y_), 1))
            
        pos1, pos2 = self.dx_lag_pos(sdt, lag)    
        meandis = np.mean((x_[pos2, :] - x_[pos1, :])**2 + (y_[pos2, :] - y_[pos1, :])**2, axis=0)
        return meandis

    def tq_autox(self, x_, y_, *argv):

        sdt, lag = argv[0][0], argv[0][1]
        
        if x_.ndim == 1:
            x_ = x_.reshape((len(x_), 1))
            y_ = y_.reshape((len(y_), 1))
            
        pos1, pos2 = self.dx_lag_pos(sdt, lag) 
        x1, x2 = x_[pos1, :], x_[pos2, :]

        a = x1 - np.expand_dims(x1.mean(axis=0), axis=0)
        b = x2 - np.expand_dims(x2.mean(axis=0), axis=0)
        autodx = np.sum(a * b, axis=0) / np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))

        return autodx

    def tq_autodis(self, x_, y_, *argv):
        
        sdt = argv[0][0]
        
        if x_.ndim == 1:
            x_ = x_.reshape((len(x_), 1))
            y_ = y_.reshape((len(y_), 1))

        pos1, pos2 = self.calautodx_pos(sdt)
        dis = np.sqrt((x_[:-1, :] - x_[1:, :])**2 + (y_[:-1, :] - y_[1:, :])**2)
        dx1, dx2 = dis[pos1, :], dis[pos2, :]

        a = dx1 - np.expand_dims(dx1.mean(axis=0), axis=0)
        b = dx2 - np.expand_dims(dx2.mean(axis=0), axis=0)
        autodx = np.sum(a * b, axis=0) / np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))

        return autodx

    # def tq_autox(self, x_, *argv): 
    #     # argv contains sdt and lag time
    
    #     if x_.ndim == 1:
    #         x_ = x_.reshape((len(x_), 1))
            
    #     sdt, lag = argv[0][0], argv[0][1]
    #     pos1, pos2 = self.dx_lag_pos(sdt, lag)
    #     x1, x2 = x_[pos1, :], x_[pos2, :]
        
    #     a = x1 - np.expand_dims(x1.mean(axis=0), axis=0)
    #     b = x2 - np.expand_dims(x2.mean(axis=0), axis=0)
    #     temp_autox = np.sum(a * b, axis=0) / np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))
        
    #     return temp_autox
            
    def tq_autodx(self, x_, *argv):
        # argv contains sdt
        
        if x_.ndim == 1:
            x_ = x_.reshape((len(x_), 1))

        pos1, pos2 = self.calautodx_pos(argv[0][0])
        sdx = (np.diff(x_, axis=0))
        dx1, dx2 = sdx[pos1, :], sdx[pos2, :]

        a = dx1 - np.expand_dims(dx1.mean(axis=0), axis=0)
        b = dx2 - np.expand_dims(dx2.mean(axis=0), axis=0)
        temp_autodx = np.sum(a * b, axis=0) / np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))

        return temp_autodx

    def p_value_disp(self, conc, test_eval, orders, a=1, b=1, sample_size=4000):

        n_test = len(test_eval)
        p_tot = np.zeros((len(orders), n_test))
        self.tot_msd[conc] = []
        
        count = 0

        for i in range(len(orders)):
            sx, sy, st = self.loadSelectTraj(conc, orders[i])
            sdx, sdy, sdt = np.diff(sx), np.diff(sy), np.diff(st)
            
            # posterior
            s = np.sum((sdx ** 2 + sdy ** 2) / (4 * sdt))
            D_samples = stats.invgamma.rvs(a=a + len(sdx), scale=b + s, size=sample_size)  

            sim_x, sim_y = base_bm_all(D_samples, int(st[-1] - st[0] + 1), st)

            tq_sim = np.zeros((sample_size, n_test))
            for j in range(n_test):
                argvs = []
                    
                for k in range(len(test_eval[j][1])):
                    argvs.append(eval(test_eval[j][1][k]))
                tq_sim[:, j] = eval(test_eval[j][0])(sim_x, sim_y, argvs)      
            
            tq_true = np.zeros((n_test, ))    
            for j in range(n_test):            

                argvs = []
                for k in range(len(test_eval[j][1])): argvs.append(eval(test_eval[j][1][k]))
                    
                tq_true[j] = eval(test_eval[j][0])(sx, sy, argvs)
            
                p_tot[i, j] = np.where((tq_true[j] <= tq_sim[:, j]) == False)[0].shape[0] / sample_size

                # if test_eval[j][0] == 'self.tq_maxdis' and p_tot[i, j] >= 0.95: 
                #     self.BI[conc][orders[i]] = eval(test_eval[j][1][1])

            self.tot_msd[conc].append([np.log(tq_true), np.log(tq_sim).mean(axis=0), np.log(tq_sim).std(axis=0)])
            self.tot_msd_id[conc][orders[i]] = count
            count += 1

        return p_tot

    # generate p_values
    def p_value_general(self, conc, test_eval, orders, a=1, b=1, sample_size=4000):

        n_test = len(test_eval)

        p_tot = np.zeros((len(orders), n_test * 2))

        for i in range(len(orders)):
            sx, sy, st = self.loadSelectTraj(conc, orders[i])
            sdx, sdy, sdt = np.diff(sx), np.diff(sy), np.diff(st)
            
            # posterior
            s = np.sum((sdx ** 2 + sdy ** 2) / (4 * sdt))
            D_samples = stats.invgamma.rvs(a=a + len(sdx), scale=b + s, size=sample_size)

            # generate posterior predictive samples
            # also simulates missing frames observed in the real dataset
            sim_x = (base_bm_all(D_samples, int(st[-1] - st[0] + 1)))[(st - st[0]).astype(int), :]
            test_quantities = np.zeros((sample_size, n_test))
            for j in range(n_test):
                argvs = []
                    
                for k in range(len(test_eval[j][1])):
                    argvs.append(eval(test_eval[j][1][k]))
                test_quantities[:, j] = eval(test_eval[j][0])(sim_x, argvs)

            # compute p-values
            for j in range(n_test):

                argvs = []
                for k in range(len(test_eval[j][1])): argvs.append(eval(test_eval[j][1][k]))

                tq_true_x = eval(test_eval[j][0])(sx, argvs)

                tq_true_y = eval(test_eval[j][0])(sy, argvs)

                p_tot[i, j * 2] = np.where((tq_true_x <= test_quantities[:, j]) == False)[0].shape[0] / sample_size
                p_tot[i, j * 2 + 1] = np.where((tq_true_y <= test_quantities[:, j]) == False)[0].shape[0] / sample_size

        return p_tot

    def rank(self, conc, sel_ind, ml, a=1, b=1):

        ranking = []
        tl, index = [], []
        map_, avg_dt = [], []
        mom = self.tot_mom[conc]
        
        for ind in sel_ind:
            sx, sy, st = self.loadSelectTraj(conc, ind)
            sdx, sdy, sdt = np.diff(sx), np.diff(sy), np.diff(st)

            if len(sx) < ml:
                continue
                
            avg_dt.append(sdt.mean())
            ranking.append(np.mean(mom[ind][:, 0]))
            tl.append(len(sx))
            index.append(ind)

            map_.append(self.MAP_bm(sdx, sdy, sdt, a=a, b=b))

        return ranking, tl, index, map_, avg_dt

    def meLogEv(self, conc, ind_):
        longx, longy, longt = self.loadSelectTraj(conc, ind_)

        lam = 2
        a, b = 1, 1

        mat = diffusing_covariance_mt(np.diff(longt))
        mat = mat + np.eye(len(longt)) * lam

        alpha = a + len(longt)
        invmat = np.linalg.inv(mat)
        x, y = longx - longx[0], longy - longy[0]
        beta = b + (x @ invmat @ x) / 4 + (y @ invmat @ y) / 4

        logev0 = a * np.log(b) + scipy.special.gammaln(alpha) - \
                 alpha * np.log(beta) - scipy.special.gammaln(a) - np.log(np.linalg.det(mat))

        logEv = [logev0]
        dif = []
        for i, lam in enumerate(np.logspace(0.23, -3, 30)):
            mat = diffusing_covariance_mt(np.diff(longt))
            mat = mat + np.eye(len(longt)) * lam

            alpha = a + len(longt)
            invmat = np.linalg.inv(mat)
            x, y = longx - longx[0], longy - longy[0]
            beta = b + (x @ invmat @ x) / 4 + (y @ invmat @ y) / 4

            logev = a * np.log(b) + scipy.special.gammaln(alpha) - \
                    alpha * np.log(beta) - scipy.special.gammaln(a) - np.log(np.linalg.det(mat))

            dif.append(logev - logEv[i])
            logEv.append(logev)

        for i in range(len(logEv)):
            logEv[i] = logEv[i] + dif[-1] * (len(logEv) - 1 - i)

        return [np.insert(np.logspace(0.23, -3, 30), 0, 2), logEv], [np.logspace(0.23, -3, 30), dif]

    def computePosteriors(self, conc, order, alpha=1, beta=1):

        mode = np.zeros((len(order), ))
        CI = np.zeros((len(order), 2))

        for i in range(len(order)):
            sx, sy, st = self.loadSelectTraj(conc, order[i])
            sdx, sdy, sdt = np.diff(sx), np.diff(sy), np.diff(st)
            alpha_ = alpha + len(sdx)
            beta_ = beta + np.sum((sdx**2 + sdy**2)/(4*sdt))
            mode[i] = beta_ / (alpha_ - 1)

            CI[i, 0] = stats.invgamma.ppf(0.025, alpha_, scale=beta_)
            CI[i, 1] = stats.invgamma.ppf(0.975, alpha_, scale=beta_)

        return mode, CI

    def computeHyperParams(self, conc, orders):

        log_tau_i = np.zeros((len(orders), ))
        a_pre, b_pre = np.zeros((len(orders), )), np.zeros((len(orders), ))

        for k, sel_ind in enumerate(orders): 
        
            # correct large jump steps
            sx, sy, st = self.loadSelectTraj(conc, sel_ind)
            sdx, sdy, sdt = np.diff(sx), np.diff(sy), np.diff(st)
            
            # compute stats for inference
            log_tau_i[k] = np.sum(np.log(1/(4*np.pi*sdt)))
            a_pre[k] = len(sdx)
            b_pre[k] = np.sum((sdx**2+sdy**2) / (4*sdt))
            
        # map estimator
        AB0 = np.array([2., 2.])
        param_ = [a_pre, b_pre, log_tau_i, 0, 1, 0, 1]
        result = minimize(logp_lnGamma, AB0, param_, method='L-BFGS-B')
        print('alpha = %f, beta = %f, optimize success = %s' %(result.x[0], result.x[1], result.success))

        return result.x


def BM_me_drift(D, t, N, sigmaTrue, driftx, drifty, seed=None): 
    
    if seed:
        np.random.seed(seed)
    
    dt = np.diff(t)
    dx = np.random.normal(driftx*dt,np.sqrt(2*D*dt))
    x = np.insert(np.cumsum(dx), 0, 0) 
    X = x + np.random.normal(0,sigmaTrue,N)
    
    dy = np.random.normal(drifty*dt,np.sqrt(2*D*dt))
    y = np.insert(np.cumsum(dy), 0, 0) 
    Y = y + np.random.normal(0,sigmaTrue,N)
    
    return X, Y


def base_bm_all(D, n_length, st):
    dt = 1
    dx = np.random.normal(0, np.sqrt(2 * D * dt), size=(n_length*2 - 2, D.shape[0]))
    x = np.insert(np.cumsum(dx[:n_length-1, :], axis=0), 0, 0, axis=0)
    y = np.insert(np.cumsum(dx[n_length-1:, :], axis=0), 0, 0, axis=0)

    return x[(st - st[0]).astype(int), :], y[(st - st[0]).astype(int), :]


def sort_by_entry(list_lis, dtype, order):
    m = np.array(list_lis[0]).reshape((len(list_lis[0]), 1))
    for i in range(len(list_lis))[1:]:
        temp_arr = np.array(list_lis[i]).reshape((len(list_lis[i]), 1))
        m = np.hstack((m, temp_arr))

    # sort based on specific entry
    temp = []
    for i in range(m.shape[0]):
        temp.append(tuple(m[i, :]))
    sorted_m = np.sort(np.array(temp, dtype=dtype), order=order).tolist()

    # modify the input list
    for i in range(len(sorted_m)):
        for j in range(len(list_lis)):
            list_lis[j][i] = sorted_m[i][j]

    del sorted_m, temp, m


def base_HPW_D(init_pos, t, D, well_pos, lambda_):
    x0, y0 = init_pos
    x_c, y_c = well_pos
    x = [x0]
    y = [y0]
    tau = np.diff(t)
    for idx in range(len(t) - 1):
        mu_x = x_c + (x[-1] - x_c) * np.exp(-lambda_ * tau)
        sd_x = np.sqrt(D / lambda_ * (1.0 - np.exp(-2.0 * lambda_ * tau)))
        x.append((np.random.normal(loc=mu_x, scale=sd_x, size=1))[0])

        mu_y = y_c + (y[-1] - y_c) * np.exp(-lambda_ * tau)
        sd_y = np.sqrt(D / lambda_ * (1.0 - np.exp(-2.0 * lambda_ * tau)))
        y.append((np.random.normal(loc=mu_y, scale=sd_y, size=1))[0])
    return np.vstack((x, y))


def shrinkage(D_unpool, D_pool, track_length):

    ind = [i for i in range(len(track_length))]
    dtype = [('log_d', float), ('index', int)]
    order = 'log_d'
    sort_by_entry([track_length, ind], dtype, order)

    b1 = np.where(np.array(track_length) <= 25)[0][-1]
    b2 = np.where(np.array(track_length) <= 50)[0][-1]
    b3 = np.where(np.array(track_length) <= 100)[0][-1]

    plt.figure(figsize=(10, 8))
    plt.scatter([i for i in range(len(track_length))], np.log(D_pool)[ind], \
                s=26, alpha=0.6, label='pool, buffer', c='b')
    plt.scatter([i for i in range(len(track_length))], np.log(D_unpool)[ind], \
                s=26, alpha=0.6, label='unpool, a=1, b=1', c='r')
    plt.ylim(-1, 1.5)
    plt.xlabel('Increase track length')
    plt.ylabel('lnD')
    plt.xticks([], " ")
    x1, x2, y1, y2 = plt.xlim()[0], plt.xlim()[1], plt.ylim()[0], plt.ylim()[1]

    plt.fill_between([0, b1], [y1, y1], [y2, y2], alpha=0.3, label='10-25')
    plt.fill_between([b1, b2], [y1, y1], [y2, y2], alpha=0.3, label='25-50')
    plt.fill_between([b2, b3], [y1, y1], [y2, y2], alpha=0.3, label='50-100')
    plt.fill_between([b3, x2], [y1, y1], [y2, y2], alpha=0.3, label='100+')

    for i in range(len(track_length)):
        plt.arrow(
            i,
            np.log(D_unpool)[ind][i],
            0,
            np.log(D_pool)[ind][i] - np.log(D_unpool)[ind][i],
            fc="k",
            ec="k",
            length_includes_head=True,
            alpha=0.6,
            head_width=0.02,
        )

    plt.legend()
    plt.show()

    return


def logp_lnGamma(ab, param): 

    a_pre, b_pre, log_tau_i, a_mu, a_scale, b_mu, b_scale = param
    
    a_i = ab[0] + a_pre
    b_i = ab[1] + b_pre

    logP = np.sum(ab[0]*np.log(ab[1]) - a_i*np.log(b_i) + scipy.special.gammaln(a_i) - \
                  scipy.special.gammaln(ab[0]) + log_tau_i)
    
    prior_a = -np.log(ab[0]*a_scale*np.sqrt(2*np.pi)) - (np.log(ab[0]) - a_mu)**2 / (2*a_scale**2)
    
    prior_b = -np.log(ab[1]*b_scale*np.sqrt(2*np.pi)) - (np.log(ab[1]) - b_mu)**2 / (2*b_scale**2)
    
    return -(logP+prior_a+prior_b)