#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:45:45 2019

@author: zhaoyu
"""

import numpy as np
import time as tm
from tqdm import tqdm

# player's initial sum of points: [12,21]
N_SUM = 10

# dealer's showing card: [ace,10]
N_SHOW = 10

# deal card
def deal(cnt=0):
    if cnt == 0:
        card = np.random.randint(low=1, high=14)
        card = min(10, card)
    else:
        card = np.random.randint(low=1, high=14, size=cnt)
        card[card>10] = 10
    return card

# Monte Carlo estimation of one policy
def play(init_state):
    '''
    for fixed initial state, requires to traverse all the states
    '''
    player_sum, dealer_show, usable_ace = init_state
    
    # deal for player
    card_player = deal(10)
    card_dealer = deal(20)
    
    cnt_player = 0
    while player_sum < 22:
        if player_sum in [20, 21]:
            break
        card = deal()
#        card = card_player[cnt_player]
#        cnt_player += 1
        player_sum += card
        if usable_ace and player_sum>=22:
            player_sum -= 10
            usable_ace = False
#        print(card, player_sum)
        if player_sum>=22:
#            print(player_sum)
            return -1
#    print(player_sum)
        
    # deal for dealer for initialization
    usable_ace_dealer = False
    if dealer_show == 1:
        dealer_show = 11
        usable_ace_dealer = True
    dealer_sum = dealer_show
    cnt_dealer = 0
    while dealer_sum < 17:
        card = deal()
#        card = card_dealer[cnt_dealer]
#        cnt_dealer += 1
        if card == 1 and dealer_sum<11:
            card = 11
            usable_ace_dealer = True
        dealer_sum += card
        if dealer_sum > 21 and usable_ace_dealer == True:
            dealer_sum -= 10
            usable_ace_dealer = False
#        print(card, dealer_sum)
        if dealer_sum > 21:
#            print(dealer_sum)
            return 1
#    print(dealer_sum)
    
    # compare player's point with dealer's
    if player_sum < dealer_sum:
        return -1
    elif player_sum == dealer_sum:
        return 0
    else:
        return 1
    
def rollout(policy, init_state=None, init_action=None):
    '''
    for each individual episode
    '''
    # record the historical player's score
    trajectory = []
    
    # initialize player's score: [12, 21]
    if init_state == None:
        player_sum = 0
        usable_ace_player = False
        while player_sum < 12:
            card = deal()
            if card == 1 and player_sum < 11:
                card = 11
                usable_ace_player = True
            player_sum += card
            if player_sum > 21 and usable_ace_player:
                player_sum -= 10
                usable_ace_player = False
                
        # initialize dealer's show card: [ace,10]
        dealer_show = deal()
    else:
        player_sum, dealer_show, usable_ace_player = init_state
        usable_ace_player = True if usable_ace_player == 1 else False

    # deal for player
    while player_sum < 22:
        if init_action == None:
#            action = int(policy[player_sum-12, dealer_show-1, int(usable_ace_player)])
            action = int(policy(player_sum, dealer_show, usable_ace_player))
        else:
            action = init_action
            init_action = None
        trajectory.append([player_sum, usable_ace_player, action])
        if action == 0:
            break
        card = deal()
        player_sum += card
#        print(card, player_sum)
        if usable_ace_player and player_sum>21:
            player_sum -= 10
            usable_ace_player = False
        if player_sum>21:
            return trajectory, dealer_show, -1

    # deal for dealer
    usable_ace_dealer = False
    dealer_sum = dealer_show
    if dealer_show == 1:
        dealer_sum = 11
        usable_ace_dealer = True
    t_dealer = [dealer_sum]
    while dealer_sum < 17:
        card = deal()
        if card == 1 and dealer_sum<11:
            card = 11
            usable_ace_dealer = True
        t_dealer.append(card)
        dealer_sum += card
        if dealer_sum > 21 and usable_ace_dealer == True:
            dealer_sum -= 10
            usable_ace_dealer = False
        if dealer_sum > 21:
            return trajectory, dealer_show, 1
        
    # compare player's score with dealer's
    assert(player_sum<22)
    assert(dealer_sum<22)
    if player_sum < dealer_sum:
        return trajectory, dealer_show, -1
    elif player_sum == dealer_sum:
        return trajectory, dealer_show, 0
    else:
        return trajectory, dealer_show, 1

def MC_on_policy_estimation(n_iter):
    policy = np.zeros([N_SUM, N_SHOW, 2], dtype=np.int32)
    policy[:8, :, :] = 1
    
    vf_usable_ace = np.zeros([N_SUM, N_SHOW])
    cnt_usable_ace = np.zeros([N_SUM, N_SHOW])
    vf_usless_ace = np.zeros([N_SUM, N_SHOW])
    cnt_usless_ace = np.zeros([N_SUM, N_SHOW])
    
    def take_action(player_sum, dealer_show, usable_ace):
        return policy[player_sum-12, dealer_show-1, int(usable_ace)]
    
    pre_time = tm.time()
    for _ in tqdm(range(n_iter)):
        trace, dealer_show, g = rollout(take_action)
        for r, usable_ace, action in trace:
            if usable_ace:
                vf_usable_ace[r-12, dealer_show-1] += g
                cnt_usable_ace[r-12, dealer_show-1] += 1
            else:
                vf_usless_ace[r-12, dealer_show-1] += g
                cnt_usless_ace[r-12, dealer_show-1] += 1
                
    for i in range(N_SUM):
        for j in range(N_SHOW):
            vf_usable_ace[i,j] = 0 if cnt_usable_ace[i,j]==0 else vf_usable_ace[i,j]/cnt_usable_ace[i,j]
            vf_usless_ace[i,j] = 0 if cnt_usless_ace[i,j]==0 else vf_usless_ace[i,j]/cnt_usless_ace[i,j]
            
#    for i in range(N_SUM):
#        for j in range(N_SHOW):
#            for _ in range(n_iter):
#                vf_usless_ace[i, j] += play([i+12, j+1, False])
#                vf_usable_ace[i, j] += play([i+12, j+1, True])
#    vf_usless_ace /= n_iter
#    vf_usable_ace /= n_iter
            
    print(tm.time()-pre_time)
    
    return vf_usable_ace, vf_usless_ace

def MC_es(n_iter):
    n_ace_status = 2
    n_action = 2
    policy = np.zeros([N_SUM, N_SHOW, n_ace_status])
    act_val = np.zeros([N_SUM, N_SHOW, n_ace_status, n_action])
    act_val_cnt = np.ones([N_SUM, N_SHOW, n_ace_status, n_action])
#    act_val_cnt = np.zeros([N_SUM, N_SHOW, n_ace_status, n_action])
    
    # initialize Policy
    policy[:8, :, :] = 1
    
    def take_action(player_sum, dealer_show, usable_ace):
        return policy[player_sum-12, dealer_show-1, int(usable_ace)]    
    
    for _ in tqdm(range(n_iter)):
        # improve the efficacy
        init_player_sum = np.random.randint(low=12, high=22)
        init_dealer_show = np.random.randint(low=1, high=11)
        init_usable_ace = np.random.randint(low=0, high=2)
        init_s = [init_player_sum,init_dealer_show,init_usable_ace]
        init_a = np.random.randint(low=0, high=2)
        
        trace, dealer_show, g = rollout(take_action, init_s, init_a)
        
        for r, usable_ace, action in trace:
            act_val[r-12, dealer_show-1, int(usable_ace), action] += g
            act_val_cnt[r-12, dealer_show-1, int(usable_ace), action] += 1
            
        v0 = act_val[:, :, :, 0] / act_val_cnt[:, :, :, 0]
        v1 = act_val[:, :, :, 1] / act_val_cnt[:, :, :, 1]        
        policy[v0>v1] = 0
        policy[v0<v1] = 1
        
#        for i in range(N_SUM):
#            for j in range(N_SHOW):
#                for k in range(n_ace_status):
#                    v0, v1 = act_val[i, j, k, :]
#                    if 0 in act_val_cnt[i, j, k, :]:
#                        continue
#                    v0 /= act_val_cnt[i,j,k,0]
#                    v1 /= act_val_cnt[i,j,k,1]
#                    policy[i,j,k] = 0 if v0>v1 else 1

    return policy

def MC_on_policy(n_iter):
    n_ace_status = 2
    n_action = 2
    policy = np.zeros([N_SUM, N_SHOW, n_ace_status])
    act_val = np.zeros([N_SUM, N_SHOW, n_ace_status, n_action])
    act_val_cnt = np.ones([N_SUM, N_SHOW, n_ace_status, n_action])
    
    # initialize Policy
    policy[:8, :, :] = 1
    
    def take_action(player_sum, dealer_show, usable_ace, epsilon=0.05):
        val = np.random.uniform(low=0.0, high=1.0)
        if val<epsilon*0.5:
            return 1 - policy[player_sum-12, dealer_show-1, int(usable_ace)]
        return policy[player_sum-12, dealer_show-1, int(usable_ace)]
    
    for _ in tqdm(range(n_iter)):
        trace, dealer_show, g = rollout(take_action)
        
        for r, usable_ace, action in trace:
            act_val[r-12, dealer_show-1, int(usable_ace), action] += g
            act_val_cnt[r-12, dealer_show-1, int(usable_ace), action] += 1
            
        v0 = act_val[:, :, :, 0] / act_val_cnt[:, :, :, 0]
        v1 = act_val[:, :, :, 1] / act_val_cnt[:, :, :, 1]        
        policy[v0>v1] = 0
        policy[v0<v1] = 1
        
    return policy

def MC_off_policy_estimation(n_iter, init_state, weighted=True):
    init_player_sum, init_dealer_sum, init_usable_ace = init_state
    
    def behaviour_policy(player_sum, dealer_sum, usable_ace):
        return np.random.randint(low=0, high=2)
    
    pi = np.zeros([N_SUM, 2])
    pi[:8, 1] = 1
    pi[8:, 0] = 1
    b = np.ones([N_SUM, 2]) * 0.5
    
    total = 0
    rho_total = 0
    for i in range(n_iter):
        trace, dealer_show, g = rollout(behaviour_policy, init_state)
        rho = 1
        for player_sum, usable_ace, action in trace:
            rho *= pi[player_sum-12, action] / b[player_sum-12, action]
        total += rho*g
        rho_total += rho
    o_mean = total / n_iter
    w_mean = total / rho_total
    return o_mean, w_mean

def MC_off_policy_variance(n_iter):
    pass

def MC_off_policy(n_iter):
    n_ace_status = 2
    n_action = 2
    policy = np.zeros([N_SUM, N_SHOW, n_ace_status])
    act_val = np.zeros([N_SUM, N_SHOW, n_ace_status, n_action])
    act_val_cnt = np.ones([N_SUM, N_SHOW, n_ace_status, n_action])
    
    # initialize Policy
    policy[:8, :, :] = 1
    
    def take_action(player_sum, dealer_sum, usable_ace):
        return np.random.randint(low=0, high=2)
    
    for _ in tqdm(range(n_iter)):
        trace, dealer_show, g = rollout(take_action)
        
        for r, usable_ace, action in trace:
            act_val[r-12, dealer_show-1, int(usable_ace), action] += g
            act_val_cnt[r-12, dealer_show-1, int(usable_ace), action] += 1
            
        v0 = act_val[:, :, :, 0] / act_val_cnt[:, :, :, 0]
        v1 = act_val[:, :, :, 1] / act_val_cnt[:, :, :, 1]        
        policy[v0>v1] = 0
        policy[v0<v1] = 1
        
    return policy
    
if __name__ == '__main__':
#    vf_0, vf_1 = MC_on_policy_estimation(1000000)
#    policy = MC_es(1000000)
#    policy = MC_on_policy(2000000)
    print(MC_off_policy_estimation(100000, [13, 2, True]))