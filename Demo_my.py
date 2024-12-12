import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import random
import pickle
from agent_20_grid.agent import Agent
# import matplotlib.pyplot as plt
import argparse
from envDispatch_v2 import Environment
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(cur_dir, 'Data_sys/')

coor = np.array([[0.00204292, -0.00564301], [0.00612177, -0.00147766], [0.0040788, 0.00416545]])
parser = argparse.ArgumentParser()
parser.add_argument("--frac", default=0.2, type=float)
parser.add_argument("--policy", default="tlookahead_v2_minimax", help='order_dispatch policy')
'''
stay, idle, tlookahead, tlookahead_v2, tlookahead_pickup, tlookahead_v2_pickup, tl_pk_reduce_tr_time, tl_v2_pk_reduce_tr_time
value_based, 5neighbor, tlookahead_v2_on_off, tlookahead_v2_on_off_v2, tlookahead_v2_adaN
tlookahead_v2_reduced, tlookahead_v2_reduced_repo_e, tlookahead_v2_minimax, tlookahead_v3, max_value_based
tlookahead_v0, neural_lp
'''
parser.add_argument("--online", default="true")
parser.add_argument("--online_transition", default="false")
parser.add_argument("--log_dir", default="temp")
parser.add_argument("--online_travel_time", default="false")
parser.add_argument("--obj", default="rate", help='rate or reward or reward_raw or reward_discount')
parser.add_argument("--obj_penalty", default=0,
                    help='reposition penalty coefficients added to the objective function')
parser.add_argument("--neighbor", default="true", help='whether using neighbor information in prediction')
parser.add_argument("--generate", default="neighbor",
                    help='generator orders using normal or neighbor setting (normal or neighbor)')
parser.add_argument("--method", default="lstm_cnn", help='prediction method')
'''
 lasso, ridge, cnn, pcr_with_ridge, pcr_with_lasso, lstm_cnn
'''
parser.add_argument("--travel_time_type", default="order", help='matrix or order (using actual travel time)')
parser.add_argument("--noiselevel", default=0, type=float)
parser.add_argument("--number_driver", default=300, type=int)
parser.add_argument("--unbalanced_factor", default=0, type=float)
parser.add_argument("--tlength", default=20, type=int,
                    help='t length of T-lookahead policy 0: no prediction otherwise using prediction')
parser.add_argument("--value_weight", default=0.2, type=float)
parser.add_argument("--collect_order_data", default="false")
parser.add_argument("--on_offline", default="false")
parser.add_argument("--start_hour", default=13, type=int)
parser.add_argument("--stop_hour", default=20, type=int)
parser.add_argument("--obj_diff_value", default=0, type=int,
                    help='whether to use difference of the value function for the objective')
parser.add_argument("--num_grids", default=20, type=int)
parser.add_argument("--make_arr_inaccurate", default="false")
parser.add_argument("--wait_minutes", default=5, type=int)
parser.add_argument("--simple_dispatch", default="true")
parser.add_argument("--split", default="true")

args = parser.parse_args()
'''
Simulator Starting time: 13:00 p.m. --- 20:00 p.m.
'''

print("---------------------------------------")
print(
    f"Policy: {args.policy}, Online: {args.online}, Online_transition: {args.online_transition}, Neighbor: {args.neighbor}, Method: {args.method}")
print("---------------------------------------")

file_name = f'neighbor_{args.neighbor}_online_ar_{args.online}_online_tr_{args.online_transition}_online_time_{args.online_travel_time}_travel_time_{args.travel_time_type}'
file_name_hour = file_name + "hour_file" + time.strftime("%Y%m%d-%H%M%S")

if not os.path.exists(
        f"./{args.log_dir}/value_{float(args.value_weight):.2f}/pent_{float(args.obj_penalty):.2f}/{args.number_driver}/{args.obj}/{args.policy}/frac_{float(args.frac):.3f}/Tlength_{args.tlength}/Noise_{args.noiselevel:.2f}"):
    os.makedirs(
        f"./{args.log_dir}/value_{float(args.value_weight):.2f}/pent_{float(args.obj_penalty):.2f}/{args.number_driver}/{args.obj}/{args.policy}/frac_{float(args.frac):.3f}/Tlength_{args.tlength}/Noise_{args.noiselevel:.2f}")


def main():
    num_driver = args.number_driver

    environment = Environment(
        args,
        num_driver=num_driver,
        driver_control=False,
        order_control=True,
        on_offline=args.on_offline,
        start_hour=args.start_hour,
        stop_hour=args.stop_hour,
        num_grids=args.num_grids,
        wait_minutes=args.wait_minutes
    )

    # Baseline: no repositioning, fixed charging rates
    if args.policy == 'baseline':
        agent = Agent(
            frac=args.frac,
            generate=args.generate,
            policy='stay',  # Fixed charging and no repositioning
            online=args.online,
            neighbor=args.neighbor,
            method=None,  # Baseline does not use predictions
            tlength=0,  # No prediction horizon
            obj='rate',  # Focus on completion rate
            num_driver=num_driver,
            obj_penalty=0,
            value_weight=0.0,  # Simplified objective weight
            simple_dispatch=True,
            split=args.split
        )
    else:
        raise ValueError("Only 'baseline' policy is supported for this configuration.")

    start_time = datetime(2016, 10, 31, 16, 1)
    start_time += timedelta(hours=args.start_hour)
    t = start_time
    t_delta = timedelta(seconds=60)

    # Initialize environment with baseline driver distribution
    num_driver_list = np.ones(24) * args.number_driver
    num_driver_list = num_driver_list.astype(int)
    driver_dist = np.ones(args.num_grids) * (1.0 / args.num_grids)
    num_order_list = np.ones(24) * 100  # Simplified order generation
    time_idx = t.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).hour

    environment.env_start(
        num_online_drivers=num_driver_list[time_idx],
        num_orders=num_order_list[time_idx],
        driver_dist=driver_dist
    )

    # Simulation
    while (t - start_time).days < 1:
        dispatch_observ = environment.generate_observation_od(num_order_list[time_idx])
        if len(dispatch_observ) > 0:
            dispatch_action = agent.dispatch(dispatch_observ)
            environment.env_update_od(dispatch_action)

        if args.policy != 'stay':
            repo_observ = environment.generate_observation_rp()
            if len(repo_observ) > 0:
                repo_action = agent.reposition(repo_observ)
                environment.env_update_rp(repo_action)

        t += t_delta
        environment.print_information()
        environment.env_update()
        environment.update_on_offline()
        environment._set_time(t.timestamp())

        total_reward = environment._get_total_reward()
        print(f"Total reward: {total_reward}")

        if (t - start_time).seconds >= (args.stop_hour - args.start_hour) * 3600 - 60:
            print("Simulation Complete")
            exit()

if __name__ == "__main__":
    args.policy = 'baseline'
    main()
