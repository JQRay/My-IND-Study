from base import BaseEnvironment
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
# import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import math
import random

REGIONCENTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'region_center.npy')

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir_20 = os.path.join(cur_dir, 'Data_sys_20_grid/')
data_dir_100 = os.path.join(cur_dir, 'Data_sys_100_grid/')

DISTANCES = os.path.join(data_dir_20, 'distances_between_grids.npy')
DISTANCES_100 = os.path.join(data_dir_100, 'distances_between_grids.npy')

coor = np.array([[0.00204292, -0.00564301], [0.00612177, -0.00147766], [0.0040788, 0.00416545]])

MIN_PER_SLOT = 10  # in minutes


class Environment(BaseEnvironment):
    def __init__(self, args, num_driver=3000, num_grids=20, charge_threshold=0.2, charge_rate=0.1, *args, **kwargs):
        super().__init__(args, num_driver, num_grids, *args, **kwargs)

        # Charging-related initialization
        self.charge_threshold = charge_threshold
        self.charge_rate = charge_rate

        # Initialize battery levels (assuming each driver starts with 100% battery)
        self.battery_level = np.ones(self.num_driver)

        # Charging station locations
        self.charge_stations = np.array([5, 10, 15])

    def check_and_charge(self):
        for i in range(self.num_driver):
            if self.battery_level[i] < self.charge_threshold:
                current_grid = self.state[i, 0]

                if current_grid in self.charge_stations:
                    self.charge_driver(i)

    def charge_driver(self, driver_id):
        if self.battery_level[driver_id] < 1.0:
            self.battery_level[driver_id] = min(1.0, self.battery_level[driver_id] + self.charge_rate)
            print(f"Driver {driver_id} is charging. Current battery level: {self.battery_level[driver_id]:.2f}")

    def update_driver_state(self):
        self.check_and_charge()
        for i in range(self.num_driver):
            if self.state[
                i, 4] == 1:
                self.battery_level[i] -= 0.01

            if self.battery_level[i] < self.charge_threshold:
                self.state[i, 3] = 1

    def env_update(self):
        self.update_driver_state()
        super().env_update()



    #our baseline method
    def reposition_vehicles(self):
        for j in range(self.num_regions):
            reposition_count = self.eta_base * (self.lambda_j[t] / sum(self.lambda_k[t]))
            self.reposition_vehicles(j, reposition_count)

    def fixed_charging_strategy(self):
        for i in range(self.num_regions):
            vehicles_to_charge = min(self.rho_base[i] * self.C_max[i], self.C_max[i] * self.s[i])
            self.charge[i] = vehicles_to_charge

    def update_energy(self):
        for i in range(self.num_regions):
            charge_rate = min(self.e_c * self.c_base[i], self.SOAC_max - self.SOAC[i])
            self.SOAC[i] += charge_rate
            self.energy[i] = charge_rate

    def calculate_objective(self):
        order_completion = sum(self.a_base[i] for i in range(self.num_regions))
        charging_costs = sum(self.e_base[i] * self.m[i] for i in range(self.num_regions))
        objective = order_completion - self.w_m * charging_costs
        return objective

    def check_flow_constraint(self):
        total_flow = sum(self.e_base[i] + self.f_base[i] for i in range(self.num_regions)) + sum(
            self.c_base[i] for i in range(self.num_regions))
        if total_flow != 1:
            print("Flow constraint violated!")





def spherical_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    return: meters
    """

    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    lon1 = np.reshape(lon1, (-1, 1))
    lon2 = np.reshape(lon2, (1, -1))
    lat1 = np.reshape(lat1, (-1, 1))
    lat2 = np.reshape(lat2, (1, -1))

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371

    return c * r * 1000


class Environment(BaseEnvironment):
    def __init__(self, args, num_driver=3000, driver_control=False, order_control=True, on_offline="false",
                 start_hour=13, stop_hour=20, num_grids=20, wait_minutes=0):
        """
        :param num_driver: int, number of drivers in total in the system
        :param on_offline: bool, If False, all drivers are online all the time. Otherwise, the drivers suffer from
        online and offline probability
        :param order_control: bool, control the mode of order generation, if True, the order will generated based on
        order_per_hour, if False, the param order_per_hour is useless.
        """
        self.num_driver = num_driver
        self.seconds = 60
        self.t_delta = timedelta(seconds=self.seconds)
        self.num_grids = num_grids

        # self.order_per_hour = None
        self.order_control = order_control
        self.driver_control = driver_control
        self.noiselevel = args.noiselevel
        self.travel_time_type = args.travel_time_type
        if on_offline == "true":
            self.on_offline = True
        else:
            self.on_offline = False
        if self.on_offline and (start_hour != 13 or stop_hour != 20):
            print('ERROR: on_offline=True and (start_hour != 13 or stop_hour != 20) is not implemented')
            exit(1)
        if self.on_offline and self.num_grids == 20:
            self.on_rate = np.load(data_dir_20 + "on_rate_3.npy")  # self.on_rate should be matrix with size (420,20)
            self.off_rate = np.load(data_dir_20 + "off_rate_3.npy")  # self.off_rate should be matrix with size (420,20)
        else:
            self.on_rate = None
            self.off_rate = None
        """
        row index is driver id
        state[:, 0] the driver's current grid or next gird
        state[:, 1] time (minutes) left to the destination if it is greater than 0
                    <=0 means it is idle
        state[:, 2] normally 1
        state[:, 3] 0 for online; 1 for offline
        state[:, 4] 0 for without passengers; 1 for with passengers
        state[:, 5] 0 means that the driver was repositioned before they become idle;
                    1 means that the driver was driving for an orders before they become idle
        """
        self.state = np.zeros([self.num_driver, 6], dtype=int)
        self.pos = np.zeros([self.num_driver, 2])  # current or next longitude and latitude
        center_info = np.load(REGIONCENTER_PATH, mmap_mode='r')
        self.grid_id_map = center_info[:, 0]  # list of strings indicating different grids
        self.grid_center_map = center_info[:, 1:3].astype(np.float)  # coordinates of the grids

        self.num_grid_map = self.grid_id_map.shape[0]  # total number of all the grids

        # self.mapped_matrix = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data/map_matrix.npy"))

        self.driver_dist_on_time = None
        self.order_geo_dist = None
        self.driver_dist = None
        self.mode = None
        self.t = None
        self.order_data = None
        self.order_data_mean = None
        self.order_data_var = None
        self.order_data_num = None
        self.update_info_od = None
        self.update_info_rp = None
        self.working_time = 0
        # self.grid_space = None
        self.total_reward = 0.0
        self.driver_num_dist_on_time = None

        self.num_order_record = np.zeros([1440, 4], dtype=int)
        # The first column is total num of orders scheduled, the second column is the number of orders fullfilled, 4th column: unavailable orders
        # self.num_online_drivers = np.zeros(1440, dtype=int)
        self.reward_record = np.zeros(1440)
        # self.transition_map = np.load('./Data/transition_map.npy')
        # self.arrival_rate = np.load('./Data/lambda_orders.npy')
        if self.num_grids == 100:
            self.transition_map = np.load(data_dir_100 + 'transition_matrix_true_%d_%d.npy' % (start_hour, stop_hour))
        else:
            self.transition_map = np.load(data_dir_20 + 'transition_matrix_true_%d_%d.npy' % (start_hour, stop_hour))
        # self.arrival_rate = np.load('./Data/lam_true.npy')
        if self.num_grids == 100:
            self.arrival_rate = np.load(data_dir_100 + 'lam_true_%d_%d.npy' % (start_hour, stop_hour))
        elif args.generate == 'neighbor':
            self.arrival_rate = np.load(data_dir_20 + 'lam_true_neighbors_%d_%d.npy' % (start_hour, stop_hour))
        else:
            self.arrival_rate = np.load(data_dir_20 + 'lam_true_%d_%d.npy' % (start_hour, stop_hour))
            print("false")
        if self.num_grids == 100:
            self.mu_inv = np.load(data_dir_100 + 'inverse_travel_time_%d_%d.npy' % (start_hour, stop_hour))
        else:
            self.mu_inv = np.load(data_dir_20 + 'inverse_travel_time_%d_%d.npy' % (start_hour, stop_hour))
        if self.num_grids == 100:
            self.area_ids = list(np.load(data_dir_100 + 'top_100_grid_ids.npy'))
        else:
            self.area_ids = [7994, 5584, 7355, 231, 3147, 3121, 8188, 3128, 1448, 3573, 6391, 3909, 60, 3735, 5347, 4962,
                             5149, 379, 701, 1977]
        self.global_time_ind = 0
        self.average_pick_up = [[] for _ in range(self.num_grids)]

        self.fuel_cost = 0.0
        if self.num_grids == 20:
            self.distances = np.load(DISTANCES)
        else:
            self.distances = np.load(DISTANCES_100)
        self.normalized_distances = self.distances / np.max(self.distances)
        self.unbalanced_factor = args.unbalanced_factor
        if self.num_grids == 20:
            self.increase_spatial_imbalance()
        self.variance_lam = self.calculate_var_lam()
        self.start_hour = start_hour
        self.stop_hour = stop_hour
        self.wait_minutes = wait_minutes

    def increase_spatial_imbalance(self):
        idx_sorted_lam = np.argsort(self.arrival_rate, axis=1)
        sorted_lam = np.take_along_axis(self.arrival_rate, idx_sorted_lam, axis=1)
        sum_of_lam = np.sum(self.arrival_rate, axis=1)
        sum_of_lam_small = np.sum(sorted_lam[:, 0:self.num_grids // 2], axis=1)
        sum_of_lam_large = np.sum(sorted_lam[:, self.num_grids // 2:self.num_grids], axis=1)
        adjustment_temp = sum_of_lam * self.unbalanced_factor
        adj_small = np.zeros((self.arrival_rate.shape[0], self.num_grids // 2))
        adj_large = np.zeros((self.arrival_rate.shape[0], self.num_grids // 2))
        adj_small[sum_of_lam_small != 0, :] = np.reshape(adjustment_temp[sum_of_lam_small != 0]
                                                         / sum_of_lam_small[sum_of_lam_small != 0],
                                                         (-1, 1)) * sorted_lam[sum_of_lam_small != 0, 0:self.num_grids // 2]
        adj_large[sum_of_lam_large != 0, :] = np.reshape(adjustment_temp[sum_of_lam_large != 0]
                                                         / sum_of_lam_large[sum_of_lam_large != 0],
                                                         (-1, 1)) * sorted_lam[sum_of_lam_large != 0, self.num_grids // 2:self.num_grids]
        rows = np.tile(np.arange(self.arrival_rate.shape[0]).reshape((-1, 1)), (1, self.num_grids // 2))
        self.arrival_rate[rows, idx_sorted_lam[:, 0:self.num_grids // 2]] = self.arrival_rate[rows, idx_sorted_lam[:, 0:self.num_grids // 2]] - adj_small
        self.arrival_rate[rows, idx_sorted_lam[:, self.num_grids // 2:self.num_grids]] = self.arrival_rate[
                                                                rows, idx_sorted_lam[:, self.num_grids // 2:self.num_grids]] + adj_large

    def calculate_var_lam(self):
        variance_lam = np.mean(np.var(self.arrival_rate, axis=1))
        print('Average spatial variance: ', variance_lam)
        return variance_lam

    def env_init(self, time=0):
        """
        Initialize order data set, and the statistic properties of the orders.
        :param time: initial time
        :return:
        """
        start_time = datetime.fromtimestamp(time)
        start_time = start_time.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8)))
        self.t = start_time
        if self.num_grids == 100:
            self.order_data = pd.read_pickle(data_dir_100 + 'order_list_100.pkl')
        else:
            self.order_data = pd.read_pickle(data_dir_20 + 'order_list.pkl')

        # print(self.order_data.Reward.max())
        # exit()
        # self.grid_space = self.order_data.Start_grid.unique()

        center_info = np.load(REGIONCENTER_PATH, mmap_mode='r')
        self.grid_id = center_info[self.area_ids, 0]
        # self.grid_center = center_info[:, 1:3].astype(np.float)
        self.grid_center = center_info[self.area_ids, 1:3].astype(np.float)
        '''
        initialize grid_id and grid_center within 20 grids
        '''
        # self.num_grid = self.grid_id.shape[0]

    def env_start(self, num_online_drivers, num_orders, driver_dist):
        """
        :param num_online_drivers: set number of online drivers for the first hour.
        :param num_of_orders: set number of orders for the first hour.
        :return:
        """
        self.total_reward = 0.0
        # str0 = 'Data/order_geo_dist.npy'
        # DRIVER_DIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), str0)
        self.order_geo_dist = driver_dist
        initial_grid = np.random.choice(len(self.area_ids), size=self.num_driver, p=self.order_geo_dist)
        initial_time = np.random.randint(-2, 2, size=self.num_driver)
        time_idx = self.t.hour
        # self.state[:, 0] = self.grid_space[initial_grid]
        self.state[:, 0] = np.array(self.area_ids)[initial_grid]  # state_0 indicates grid_ids
        self.state[:, 1] = initial_time
        # self.order_per_hour = num_orders
        self.state[self.state[:, 1] > 0, 4] = 1
        self.state[:, 5] = 1

        if self.driver_control:
            num_idle_drivers = num_online_drivers
            self.state[:num_idle_drivers, 2] = 1
        else:
            self.state[:, 2] = 1
        generate_pos_idx = np.where((self.state[:, 2] == 1) & (self.state[:, 3] == 0))[0]
        self.generate_pos(generate_pos_idx)
        # observation, self.update_info_od = self.generate_observation_od()
        # self.t += self.t_delta
        return None

    def generate_orders_from_arrival(self, arrival_list):
        num_order = np.zeros(len(self.area_ids))
        for i in range(len(self.area_ids)):
            # num_order[i] = int(np.random.poisson(arrival_list[i] * (10 / MIN_PER_SLOT)) / (600 / self.seconds))
            num_order[i] = int(np.random.poisson(arrival_list[i] / MIN_PER_SLOT))
            noise = np.random.normal(0, self.noiselevel, 1)
            num_order[i] = max(num_order[i] + int(noise), 0)
        return num_order

    def num_online_cars(self):
        candidates_idx = np.where((self.state[:, 2] == 1)
                                  & (self.state[:, 3] == 0))[0]
        return candidates_idx.shape[0]

    def update_on_offline(self):
        if (not self.on_offline) or (self.num_grids != 20):
            return
        else:
            time_minute = (int(self.t.hour) - self.start_hour) * 60 + self.t.minute
            for grid_idx in range(len(self.area_ids)):
                on_diff = self.on_rate[time_minute, grid_idx] - self.off_rate[time_minute, grid_idx]
                if on_diff < 0:
                    candidates_idx = np.where((self.state[:, 0] == self.area_ids[grid_idx])
                                              & (self.state[:, 1] <= 0)
                                              & (self.state[:, 2] == 1)
                                              & (self.state[:, 3] == 0))[0]
                    if candidates_idx.shape[0] > 0:
                        off_idx = np.random.choice(candidates_idx.shape[0],
                                                   size=int(min(candidates_idx.shape[0], -on_diff)))
                        self.state[candidates_idx[off_idx], 3] = 1
                elif on_diff > 0:
                    num_on_cars = int(on_diff)
                    self.state = np.resize(self.state, (self.state.shape[0] + num_on_cars, self.state.shape[1]))
                    self.pos = np.resize(self.pos, (self.pos.shape[0] + num_on_cars, self.pos.shape[1]))
                    temp_idx = self.state.shape[0] - num_on_cars
                    for i in range(num_on_cars):
                        self.state[temp_idx + i, 0] = self.area_ids[grid_idx]
                        self.state[temp_idx + i, 1] = 0
                        self.state[temp_idx + i, 2] = 1
                        self.state[temp_idx + i, 3] = 0
                        self.state[temp_idx + i, 4] = 0
                        self.state[temp_idx + i, 5] = 1
                        self.generate_pos([temp_idx + i])

            return

    def generate_observation_od(self, num_arrival_orders):
        """
        Generate an observation for dispatching
        :return:
        """
        time_idx = self.t.hour
        record_time_idx = self.t.hour * 60 + self.t.minute

        # time_slot_arrival = (self.t.hour - 13) * 30 + self.t.minute // 2
        time_slot = (int(self.t.hour)) * 60 // MIN_PER_SLOT + self.t.minute // MIN_PER_SLOT  # index start from 0
        time_slot_arrival = (int(
            self.t.hour) - self.start_hour) * 60 // MIN_PER_SLOT + self.t.minute // MIN_PER_SLOT  # index start from 0

        if not self.order_control:
            num_order = max(int(np.random.poisson(self.order_data_mean[time_idx] / 60 / 60)), 0)
        else:
            num_order = self.generate_orders_from_arrival(self.arrival_rate[time_slot_arrival])
            # num_order = max(int(np.random.poisson(self.order_per_hour/60/30)), 0)

        # self.num_order_record[record_time_idx, 0] += num_order
        self.num_order_record[record_time_idx, 0] += num_order.sum()
        observation = []
        update_info = []
        # order_candidate = self.order_data.index[self.order_data['Start_Timeslot']//6 == time_idx].tolist()
        # order_candidate = self.order_data[self.order_data['Start_Timeslot']//6 == time_idx]
        order_candidate = self.order_data[self.order_data['Start_Timeslot'] >= time_slot]
        order_candidate = order_candidate[order_candidate['Start_Timeslot'] < time_slot + 1]
        order_idx = []
        for ii in range(len(num_order)):
            st_grid_ind = self.area_ids[ii]
            order_candidate_ = order_candidate[order_candidate['Start_grid'] == int(st_grid_ind)]
            dest_lists = np.random.choice(len(num_order), int(num_order[ii]),
                                          p=self.transition_map[time_slot_arrival][ii])
            unique, counts = np.unique(dest_lists, return_counts=True)

            possible_orders = dict(zip(unique, counts))
            for a, b in enumerate(possible_orders):
                orders_ = order_candidate_.index[order_candidate_['Drop_grid'] == self.area_ids[int(b)]].tolist()
                if len(orders_) == 0:
                    continue
                orders_pick = np.random.choice(orders_, size=int(possible_orders[b]))
                order_idx.extend(orders_pick)
                '''
                if len(orders_pick) == 1:
                    order_idx.append(orders_pick)
                else:
                    order_idx.extend(orders_pick)
                '''
        for ii in range(len(order_idx)):
            idx = order_idx[ii]
            ts = self.t.timestamp()
            osl = [self.order_data.Pick_Long[idx], self.order_data.Pick_Lati[idx]]
            ofl = [self.order_data.Drop_Long[idx], self.order_data.Drop_Lati[idx]]
            # order_time_consumption = (datetime.fromtimestamp(self.order_data.Stop_time[idx]) -
            #                          datetime.fromtimestamp(self.order_data.Start_time[idx])).seconds
            sgrid = self.order_data.Start_grid[idx]
            dgrid = self.order_data.Drop_grid[idx]
            s_idx = self.area_ids.index(int(sgrid))
            d_idx = self.area_ids.index(int(dgrid))
            mu_inv = self.mu_inv[time_slot_arrival]
            if self.travel_time_type == 'matrix':
                order_time_consumption = (1.0 / mu_inv[int(s_idx)][
                    d_idx]) * MIN_PER_SLOT * 60 + np.random.rand() * 200 - 100
            else:
                order_time_consumption = (datetime.fromtimestamp(self.order_data.Stop_time[idx]) -
                                          datetime.fromtimestamp(self.order_data.Start_time[idx])).seconds
            reward = self.order_data.Reward[idx]
            # candidate_grid_list = map_output[ii][3].copy()
            # candidate_grid_list.append(map_output[ii][0])
            order_driver_distance = spherical_distance(osl[0], osl[1], self.pos[:, 0], self.pos[:, 1]).squeeze()
            available_idx = np.where((
                                      (self.state[:, 1] <= 0)
                                      & (self.state[:, 2] == 1)
                                      & (order_driver_distance <= 1200)
                                      & (self.state[:, 4] == 0)
                                      & (self.state[:, 3] == 0)
                                     ) |
                                     (
                                      (self.state[:, 2] == 1)
                                      & (self.state[:, 0] == self.order_data.Start_grid[idx])
                                      & (self.state[:, 4] == 0)
                                      & (self.state[:, 3] == 0)
                                     )
                                     )[0]
            if len(available_idx) == 0:
                self.num_order_record[record_time_idx, 3] += 1

            # dest_diff = self.grid_center - np.array(ofl)
            # dest_dist = np.sum(np.abs(dest_diff)**2, axis=-1)
            # dest_idx = np.argmin(dest_dist)

            for dd in available_idx:
                ob = {}
                ob['order_id'] = ii
                ob['driver_id'] = dd
                ob['start_time_slot'] = time_slot_arrival
                # order_driver_distance = spherical_distance(osl[0], osl[1], self.pos[dd, 0], self.pos[dd, 1])
                ob['order_driver_distance'] = order_driver_distance[dd]
                ob['timestamp'] = self.t.timestamp()
                ob['pick_up_eta'] = order_driver_distance[dd] / 3
                ob['driver_location'] = list(self.pos[dd, :])
                ob['order_start_location'] = osl
                ob['start_id'] = sgrid
                ob['destination_id'] = dgrid
                ob['order_finish_location'] = ofl
                end_time = self.t + timedelta(seconds=int(order_driver_distance[dd] / 3) + order_time_consumption)
                ob['order_finish_timestamp'] = end_time.timestamp()
                ob['reward_units'] = reward
                ob['stop_time_slot'] = (int(
                    end_time.hour) - self.start_hour) * 60 // MIN_PER_SLOT + end_time.minute // MIN_PER_SLOT
                observation.append(ob)

                self.average_pick_up[s_idx].append(order_driver_distance[dd] / 3.)
                info = {'order_id': ii, 'driver_id': dd}
                info_dest = self.order_data.Drop_grid[idx]
                if info_dest in self.area_ids:
                    info['dest'] = self.order_data.Drop_grid[idx]
                else:
                    print("Error")
                info['reward'] = reward
                info['dest_location'] = ofl
                if order_driver_distance[dd] // 180 <= 9:
                    order_driver_distance_level = int(order_driver_distance[dd] // 180)
                    cancel_probs = self.order_data.Cancel_Prob[idx].split(",")
                    info['cancel_prob'] = float(cancel_probs[order_driver_distance_level])
                else:
                    info['cancel_prob'] = 0
                # info['time_consumption'] = order_time_consumption // 60
                info['time_consumption'] = order_time_consumption // 60 + (order_driver_distance[dd] / 3) // 60
                update_info.append(info)

        self.update_info_od = update_info

        return observation

    def env_update_od(self, action):
        """
        :param action:  the action of order dispatching
        :return: None
        """
        driver_list = []
        record_time_idx = self.t.hour * 60 + self.t.minute
        for aa in action:
            order_id = aa['order_id']
            driver_id = aa['driver_id']
            order_driver_pair = [item for item in self.update_info_od if
                                 item['order_id'] == order_id and item['driver_id'] == driver_id]
            if len(order_driver_pair) > 1 or driver_id in driver_list or self.state[driver_id, 2] == 0 or self.state[
               driver_id, 3] == 1 or self.state[driver_id, 4] == 1:
                print("Error")
                continue
            elif len(order_driver_pair) == 0:
                continue
            driver_list.append(driver_id)
            info = order_driver_pair[0]
            # current_reward = info['reward'] if random.random() > info['cancel_prob'] else 0
            current_reward = info['reward']
            self.total_reward += current_reward
            self.state[driver_id, :2] = [info['dest'], info['time_consumption']]
            self.state[driver_id, 4] = 1
            self.state[driver_id, 5] = 1
            self.pos[driver_id] = info['dest_location']
            self.working_time += float(info['time_consumption'])
            # self.pos[driver_id] = info['dest_location']
            self.num_order_record[record_time_idx, 1] += 1
            self.num_order_record[record_time_idx, 2] += 1 if current_reward != 0 else 0
            self.reward_record[record_time_idx] += current_reward

        # observation, self.update_info_od = self.generate_observation_od()
        # self.t += self.t_delta
        # if self.t.second == 0:
        #     self.env_update()
        return None


    def generate_observation_rp(self):
        observation = {}
        observation['timestamp'] = self.t.timestamp()
        observation['driver_info'] = []
        # update_info = []
        # num_driver = self.num_driver

        # When the driver was repositioned before they are idle,
        # they have to wait for wait_minutes minutes to become eligible for repositioning;
        # When the driver drove for an order before they are idle,
        # they do not need to wait to become eligible for repositioning.
        available_idx = np.where((
                                  (self.state[:, 1] <= -self.wait_minutes)
                                  & (self.state[:, 5] == 0)
                                  & (self.state[:, 2] == 1)
                                  & (self.state[:, 3] == 0)
                                 ) |
                                 (
                                  (self.state[:, 1] <= 0)
                                  & (self.state[:, 5] == 1)
                                  & (self.state[:, 2] == 1)
                                  & (self.state[:, 3] == 0)
                                 )
                                 )[0]

        #  What is the following code doing: never mind
        """
        if len(available_idx) > num_driver:
            # num_driver = len(available_idx)
            available_state = self.state[available_idx, 1]
            driver_id_list = available_idx[np.argpartition(available_state, num_driver)[0:num_driver]]
        else:
            driver_id_list = available_idx
        """
        driver_id_list = available_idx
        # start_list = []
        for driver_id in driver_id_list:
            # if np.random.rand() >= 0.2:
            #    continue
            ob = {}
            # ob['grid_id'] = self.grid_id[np.where(self.grid_space==self.state[driver_id, 0])[0]]
            grid_idx = np.where(self.area_ids == self.state[driver_id, 0])[0]
            ob['grid_id'] = self.grid_id[int(grid_idx)]
            # ob['grid_id'] = self.grid_id[np.where(self.grid_space==self.state[driver_id, 0])[0]]
            # print(self.grid_id) % str list
            # print(self.state[driver_id,0]) %id
            # print(self.grid_space) %id list
            # start_list.append(self.state[driver_id, 0])
            ob['driver_id'] = driver_id
            # print(ob)
            # print(self.grid_id)
            # print(grid_idx)
            # print(self.state[driver_id])
            # exit()
            observation['driver_info'].append(ob)

        self.update_info_rp = driver_id_list

        # self.t += self.t_delta  # Why do we update time here: do not update here, let it be updated by the user

        return observation

    def env_update_rp(self, action):
        driver_list = []
        for aa in action:
            driver_id = aa['driver_id']
            destination_id = aa['destination']

            assert destination_id in self.grid_id, "Error: Wrong grid ID."
            assert driver_id in self.update_info_rp, "Error: Wrong Driver ID."
            assert driver_id not in driver_list, "Error: Reposition driver: " + str(driver_id) + " repeatedly."
            # if driver_id not in self.update_info_rp or driver_id in driver_list:
            #     print("Error")
            #     continue
            driver_list.append(driver_id)
            s_id = self.state[driver_id, 0]
            s_idx = self.area_ids.index(s_id)
            # start_lon, start_lat = self.grid_center[s_idx, :]
            # start_lon, start_lat = self.pos[driver_id]

            dest_idxx = np.where(self.grid_id == destination_id)[0][0]
            dest_idx = self.area_ids[dest_idxx]
            # dest_lon, dest_lat = self.grid_center_map[dest_idx]
            # print(dest_idx,destination_id,self.grid_id,dest_lon,dest_lat)
            # exit()
            # dest_lon, dest_lat = self.grid_center_map[dest_idx]

            time_slot_arrival = (int(
                self.t.hour) - self.start_hour) * 60 // MIN_PER_SLOT + self.t.minute // MIN_PER_SLOT  # index start from 0
            time_slot_arrival = np.min([time_slot_arrival, (self.stop_hour - self.start_hour) * 60 // MIN_PER_SLOT - 1])
            mu_inv = self.mu_inv[time_slot_arrival]
            time_consumption = (1.0 / mu_inv[int(s_idx)][dest_idxx]) * MIN_PER_SLOT * 60 + np.random.rand() * 120 - 60
            time_consumption = time_consumption // 60
            # distance = spherical_distance(start_lon, start_lat, dest_lon, dest_lat)
            # time_consumption = int(distance/3/60)
            if self.state[driver_id, 0] != dest_idx:
                self.state[driver_id, :2] = [dest_idx, time_consumption]
                self.pos[driver_id, :] = self.grid_center_map[dest_idx, :]
                self.state[driver_id, 4] = 0
                self.state[driver_id, 5] = 0

            self.fuel_cost = self.fuel_cost + self.normalized_distances[s_idx, dest_idxx]

        # if self.t.second == 0:
        #     self.env_update()
        return None

    def env_update(self):
        # self.generate_pos()
        available_idx = np.where((self.state[:, 2] == 1) & (self.state[:, 3] == 0))[0]
        self.state[available_idx, 1] = self.state[available_idx, 1] - 1
        self.state[self.state[available_idx, 1] <= 0, 4] = 0
        print(" ")

    def env_message(self, message):
        print(" ")



    def generate_pos(self, available_idx):
        # available_idx = np.where((self.state[:, 2] == 1) & (self.state[:, 3] == 0) & (self.state[:, 4] == 0))[0]
        center = self.grid_center_map[self.state[available_idx, 0], :]
        coor_idx = np.random.randint(3, size=len(available_idx))
        coor1 = coor[coor_idx]
        coor2 = coor[np.mod((coor_idx + 1), 3)]
        noise_len = np.random.random([2, len(available_idx)])
        noise = np.transpose(
            np.multiply(np.transpose(coor1), noise_len[0, :]) + np.multiply(np.transpose(coor2), noise_len[1, :]))
        self.pos[available_idx, :] = center + noise

    def _get_total_reward(self):
        return self.total_reward

    def _set_time(self, time):
        input_time = datetime.fromtimestamp(time)
        input_time = input_time.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8)))
        self.t = input_time

    def print_information(self):
        record_time_idx = self.t.hour * 60 + self.t.minute
        print(str(self.num_order_record[record_time_idx, 0]) + " orders in this minute.")
        print(str(self.num_order_record[record_time_idx, 3]) + " unavailable, "
              + str(self.num_order_record[record_time_idx, 1]) + " scheduled.    " +
              str(self.num_order_record[record_time_idx, 1] - self.num_order_record[record_time_idx, 2])
              + " cancelled")
        if self.num_order_record[record_time_idx, 0] == 0:
            return
        else:
            print("Completion rate:" + str(self.num_order_record[record_time_idx, 2]
                                           / self.num_order_record[record_time_idx, 0])
                  + ".    Reward in this minute:" + str(self.reward_record[record_time_idx]))

    def plot_information(self):
        fullfilled_rate = self.num_order_record[:, 2] / self.num_order_record[:, 0]
        dispatched_rate = self.num_order_record[:, 1] / self.num_order_record[:, 0]
        # plt.figure(1)
        # plt.plot(fullfilled_rate, 'y.')
        # plt.plot(dispatched_rate, 'b.')
        # plt.show()
