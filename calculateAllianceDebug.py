# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:42:30 2018

@author: Ithier
"""

import pandas as pd
import numpy as np
from functools import reduce
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.from_csv.html
event_key = '2017txlu'
matches_file = event_key + '_matches.csv'
stats_file = event_key + '_stats.csv'
opr_file = event_key + '_opr.csv'
experience_file = event_key + '_rookieyear.csv'
matches = pd.DataFrame.from_csv(matches_file)
stats = pd.DataFrame.from_csv(stats_file)
experience = pd.DataFrame.from_csv(experience_file)
opr = pd.DataFrame.from_csv(opr_file)
my_dfs = [experience, stats, opr]
teams = reduce(lambda x,y: pd.merge(x,y, on = "team"), my_dfs)


def calculateAlliance(row, teams):
    # blue stats
    blue = teams.loc[(teams["team"] == row[0]) | (teams["team"] == row[1]) | (teams["team"] == row[2])]
    b_score = row[3]
    blue = blue.copy()
    blue.drop(["team"], axis = 1, inplace = True)
    sum_blue = blue.apply(np.sum, axis = 0)
    b_experience = sum_blue["experience"]
    b_wins = sum_blue["wins"]
    b_losses = sum_blue["losses"]
    b_ties = sum_blue["ties"]
    b_points = sum_blue["points"]
    b_auto = sum_blue["auto"]
    b_rotor = sum_blue["rotor"]
    b_touchpad = sum_blue["touchpad"]
    b_pressure = sum_blue["pressure"]
    
    # Red stats
    red = teams.loc[(teams["team"] == row[0]) | (teams["team"] == row[1]) | (teams["team"] == row[2])]
    r_score = row[7]
    red = blue.copy()
    red.drop(["team"], axis = 1, inplace = True)
    sum_red = red.apply(np.sum, axis = 0)
    r_experience = sum_red["experience"]
    r_wins = sum_red["wins"]
    r_losses = sum_red["losses"]
    r_ties = sum_red["ties"]
    r_points = sum_red["points"]
    r_auto = sum_red["auto"]
    r_rotor = sum_red["rotor"]
    r_touchpad = sum_red["touchpad"]
    r_pressure = sum_red["pressure"]
    return pd.Series([b_experience, b_wins, b_losses, b_ties, b_points, b_auto, b_rotor, b_touchpad, b_pressure, b_score, r_experience, r_wins, r_losses, r_ties, r_points, r_auto, r_rotor, r_touchpad, r_pressure, r_score])

df = matches.apply(lambda row: calculateAlliance(row, teams), axis = 1)
#df.columns = ["b_experience", "b_wins", "b_losses", "b_ties", "b_points", "b_auto", "b_rotor", "b_touchpad", "b_pressure", "b_score"]
