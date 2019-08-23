import argparse
import math
import numpy as np

from . import load_all
from . import show
import matplotlib.pyplot as plt


    

def compute_velocity_interaction(path, neigh_path, time_param=(9, 21, 9, 3)):
    ## Computes the angle between velocity of neighbours and velocity of pp

    T_OBS, T_SEQ, T_INT, T_STR = time_param 

    prim_vel = path[T_INT:T_SEQ] - path[T_INT-T_STR:T_SEQ-T_STR]
    theta1 = np.arctan2(prim_vel[:,1], prim_vel[:,0])
    neigh_vel = neigh_path[T_INT:T_SEQ] - neigh_path[T_INT-T_STR:T_SEQ-T_STR]
    vel_interaction = np.zeros(neigh_vel.shape[0:2])
    sign_interaction = np.zeros(neigh_vel.shape[0:2])

    for n in range(neigh_vel.shape[1]):
        theta2 = np.arctan2(neigh_vel[:, n, 1], neigh_vel[:, n, 0])
        theta_diff = (theta2 - theta1) * 180 / np.pi
        theta_diff = (theta_diff - 180) % 360
        theta_sign = theta_diff > 180
        # sign_interaction[:, n] = np.sign(theta2 - theta1 - np.pi)
        # vel_interaction[:, n] = np.abs((theta2 - theta1 - np.pi)* 180 / np.pi)
        sign_interaction[:, n] = theta_sign
        vel_interaction[:, n] = theta_diff       
    return vel_interaction, sign_interaction


def compute_theta_interaction(path, neigh_path, time_param=(9, 21, 9, 3)):
    ## Computes the angle between line joining pp to neighbours and velocity of pp

    T_OBS, T_SEQ, T_INT, T_STR = time_param

    prim_vel = path[T_INT:T_SEQ] - path[T_INT-T_STR:T_SEQ-T_STR]
    theta1 = np.arctan2(prim_vel[:,1], prim_vel[:,0])
    rel_dist = neigh_path[T_INT:T_SEQ] - path[T_INT:T_SEQ][:, np.newaxis, :]
    theta_interaction = np.zeros(rel_dist.shape[0:2])
    sign_interaction = np.zeros(rel_dist.shape[0:2])

    for n in range(rel_dist.shape[1]):
        theta2 = np.arctan2(rel_dist[:, n, 1], rel_dist[:, n, 0])
        theta_diff = (theta2 - theta1) * 180 / np.pi
        theta_diff = theta_diff % 360
        theta_sign = theta_diff > 180
        # print("MaxMin: ", np.nanmax(theta_diff), np.nanmin(theta_diff))
        # sign_interaction[:, n] = np.sign(theta2 - theta1)
        # theta_interaction[:, n] = np.abs((theta2 - theta1)* 180 / np.pi)
        sign_interaction[:, n] = theta_sign
        theta_interaction[:, n] = theta_diff
    return theta_interaction, sign_interaction

def compute_dist_rel(path, neigh_path, time_param=(9, 21, 9, 3)):
    ## Distance between pp and neighbour 
    ## Output Shape: T_pred x Number_of_Neighbours
    T_OBS, T_SEQ, T_INT, T_STR = time_param
    dist_rel = np.linalg.norm((neigh_path[T_INT:T_SEQ] - path[T_INT:T_SEQ][:, np.newaxis, :]), axis=2)
    return dist_rel


def compute_interaction(theta_rel, dist_rel, angle, dist_thresh, angle_range):
    ## Interaction is defined as 
    ## 1. distance < threshold and 
    ## 2. angle between velocity of pp and line joining pp to neighbours
    
    # theta_bool = (theta_rel < angle)
    # dist_bool = (dist_rel < dist_thresh)
    angle_low = (angle - angle_range) 
    angle_high = (angle + angle_range) 
    if (angle - angle_range) < 0 :
        theta_rel[np.where(theta_rel > 180)] = theta_rel[np.where(theta_rel > 180)] - 360
    # print(theta_rel != nan)
    # print("Low: ", angle_low)
    # print("High: ", angle_high)
    # print("Theta: ", theta_rel)
    interaction_matrix = (angle_low < theta_rel) & (theta_rel <= angle_high) & (dist_rel < dist_thresh) & (theta_rel < 500) == 1
    # print("interaction_matrix", interaction_matrix)
    return interaction_matrix


def compute_theta_vr(path):
    ## Computes the angle between velocity of pp at t_obs and velocity of pp at t_pred
    
    row1, row2, row3, row4 = path[5], path[8], path[17], path[20]
    diff1 = np.array([row2[0] - row1[0], row2[1] - row1[1]])
    diff2 = np.array([row4[0] - row3[0], row4[1] - row3[1]])
    theta1 = np.arctan2(diff1[1], diff1[0])
    theta2 = np.arctan2(diff2[1], diff2[0])
    vr1 = np.linalg.norm(diff1) / (3 * 0.4)
    vr2 = np.linalg.norm(diff2) / (3 * 0.4)
    if vr1 < 0.1:
        return 0, 0
    return theta2 - theta1, vr2


def get_interaction_matrix(rows, args, output='all'):
    ## Computes the angle between velocity of pp at t_obs and velocity of pp at t_pred
    
    ## Extract Args:
    pos_angle, pos_range, vel_angle, vel_range, dist_thresh, choice = \
    args.pos_angle, args.pos_range, args.vel_angle, args.vel_range, args.dist_thresh, args.choice

    path = rows[:, 0]
    neigh_path = rows[:, 1:]
    theta_interaction, sign_interaction = compute_theta_interaction(path, neigh_path)
    vel_interaction, sign_vel_interaction = compute_velocity_interaction(path, neigh_path)
    dist_rel = compute_dist_rel(path, neigh_path)
    
    ## str choice
    if choice == 'pos':
        interaction_matrix = compute_interaction(theta_interaction, dist_rel, pos_angle, dist_thresh, pos_range)
        chosen_interaction = theta_interaction
        # angle_thres = pos_angle

    elif choice == 'vel':
        interaction_matrix = compute_interaction(vel_interaction, dist_rel, vel_angle, dist_thresh, vel_range)
        chosen_interaction = vel_interaction
        sign_interaction = sign_vel_interaction
        # angle_thres = vel_angle

    elif choice == 'bothpos':
        interaction_matrix = compute_interaction(theta_interaction, dist_rel, pos_angle, dist_thresh, pos_range) \
                             & compute_interaction(vel_interaction, dist_rel, vel_angle, dist_thresh, vel_range)
        chosen_interaction = theta_interaction
        
    elif choice == 'bothvel':  
        interaction_matrix = compute_interaction(theta_interaction, dist_rel, pos_angle, dist_thresh, pos_range) \
                             & compute_interaction(vel_interaction, dist_rel, vel_angle, dist_thresh, vel_range)  
        chosen_interaction = vel_interaction
        sign_interaction = sign_vel_interaction
    else:
        raise NotImplementedError 

    chosen_true = chosen_interaction[interaction_matrix]
    sign_true = sign_interaction[interaction_matrix]
    dist_true = dist_rel[interaction_matrix]

    ## output choice
    if output == 'matrix':
        return interaction_matrix
    elif output == 'all':
        return interaction_matrix, chosen_true, sign_true, dist_true
    else:
        raise NotImplementedError 

def check_group(rows, dist_thresh=0.8, std_thresh=0.1):
    path = rows[:, 0]
    neigh_path = rows[:, 1:]
    # dist_rel = compute_dist_rel(path, neigh_path)
    dist_rel = np.linalg.norm((neigh_path - path[:, np.newaxis, :]), axis=2)

    mean_dist = np.mean(dist_rel, axis=0)
    # print("Mean Dist Shape: ", mean_dist.shape)
    std_dist = np.std(dist_rel, axis=0)
    # print(std_dist.shape)

    group_matrix = (mean_dist < dist_thresh) & (std_dist < std_thresh)
    # print("Group Matrix: ", group_matrix)

    group = neigh_path[:, group_matrix, :]
    return path, group, np.any(group_matrix)