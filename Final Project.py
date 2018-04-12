# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:32:00 2018

@author: Ithier
"""

from functools import reduce
import requests
import numpy as np
import pandas as pd
# https://github.com/frc1418/tbapy/blob/master/tbapy/main.py
base_url= "https://www.thebluealliance.com/api/v3/"
auth_key = 'fge7icVbwIkRUkYKFb7Bj045jGELlWspOnCTxJnhkC9jqiLRjE0VBR4ACcez4vxo'

# https://www.dataquest.io/blog/python-api-tutorial/
# https://www.digitalocean.com/community/tutorials/how-to-use-web-apis-in-python-3
# https://www.dataquest.io/blog/python-api-tutorial/
# https://www.dataquest.io/blog/python-api-tutorial/
headers = {"X-TBA-Auth-Key": auth_key}

# Get a list of the event keys we are going to go through
url = base_url + 'events/2017'
response = requests.get(url, headers = headers)
data = response.json()
event_keys = []
for i in range(0,len(data)):
    event_keys.append(data[i]["key"])
event_keys = ["2017txlu"]
# Get the number of years the team has existed
def get_experience(event_key, headers):
    """ 
        @param event_key: the event we are interested in
        @param headers: the authorizations key
        @return df_rookie_year: a dataframe with the teams and the number of years each team has existed
    """
    url = base_url + 'event/' + event_key + '/teams'
    response = requests.get(url, headers = headers)
    data = response.json()

    # create a dataframe of the team and rookie year
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html
    df_rookieyear = pd.DataFrame(columns = ["team", "experience"])
    for i in range(0, len(data)):
        name = data[i]["key"]
        year = 2017 - data[i]["rookie_year"]
        df_rookieyear = df_rookieyear.append({"team": name, "experience": year}, ignore_index = True)
    return df_rookieyear

# Get stats on the team like number of wins and losses
def get_stats(event_key, headers):
    """ 
        @param event_key: the event we are interested in
        @param headers: the authorizations key
        @return df_stats: a dataframe with the teams and stats for that team like number of wins, losses, ties, etc
    """
    url = base_url + 'event/' + event_key + '/rankings'
    response = requests.get(url, headers = headers)
    data = response.json()
    rankings = data["rankings"]
    df_stats = pd.DataFrame(columns = ["team", "rank", "wins", "losses", "ties", "points", "auto", "rotor", "touchpad", "pressure"])
    for i in range(0, len(rankings)):
        team = rankings[i]["team_key"]
        rank = rankings[i]["rank"]
        wins = rankings[i]["record"]["wins"]
        losses = rankings[i]["record"]["losses"]
        ties = rankings[i]["record"]["ties"]
        points = rankings[i]["sort_orders"][1]
        auto = rankings[i]["sort_orders"][2]
        rotor = rankings[i]["sort_orders"][3]
        touchpad = rankings[i]["sort_orders"][4]
        pressure = rankings[i]["sort_orders"][5]
        df_stats = df_stats.append({"team": team, "rank": rank, "wins": wins, "losses": losses, "ties": ties, "points": points, "auto": auto, "rotor": rotor, "touchpad": touchpad, "pressure": pressure}, ignore_index = True)
    return df_stats
# Get the match data for the competition
def get_matches(event_key, headers):
    """ 
        @param event_key: the event we are interested in
        @param headers: the authorizations key
        @return df_matches: a dataframe with the blue alliance teams and score and the red alliance team and score
    """
    url = base_url + 'event/' + event_key + '/matches/simple'
    response = requests.get(url, headers = headers)
    data = response.json()
    df_matches = pd.DataFrame(columns = ["b1", "b2", "b3", "bscore", "r1", "r2", "r3", "rscore", "level"])
    for i in range(0, len(data)):
        b1 = data[i]["alliances"]["blue"]["team_keys"][0]
        b2 = data[i]["alliances"]["blue"]["team_keys"][1]
        b3 = data[i]["alliances"]["blue"]["team_keys"][2]
        bscore = data[i]["alliances"]["blue"]["score"]
        r1 = data[i]["alliances"]["red"]["team_keys"][0]
        r2 = data[i]["alliances"]["red"]["team_keys"][1]
        r3 = data[i]["alliances"]["red"]["team_keys"][2]
        rscore = data[i]["alliances"]["red"]["score"]
        level = data[i]["comp_level"]
        df_matches = df_matches.append({"b1": b1, "b2": b2, "b3": b3, "bscore": bscore, "r1": r1, "r2": r2, "r3": r3, "rscore": rscore, "level": level}, ignore_index = True)
    return df_matches
# Create alliance info like average rank, average OPR, total number of wins, etc
def alliance(row, teams):
    """ 
        @param row: row from lambda fn using apply
        @param teams: dataframe with team statistic information
        @return : series with information of each alliance
    """
    # blue stats
    blue = teams.loc[(teams["team"] == row[0]) | (teams["team"] == row[1]) | (teams["team"] == row[2])]
    b_score = row[3]
    blue = blue.copy()
    blue.drop(["team"], axis = 1, inplace = True)
    # get the sum of things we want
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
    # get the average of things we want
    avg_blue = blue.apply(np.mean, axis = 0)
    b_rank = avg_blue["rank"]
    b_OPR = avg_blue["OPR"]
    b_DPR = avg_blue["DPR"]
    b_CCWM = avg_blue["CCWM"]
    
    # Red stats
    red = teams.loc[(teams["team"] == row[0]) | (teams["team"] == row[1]) | (teams["team"] == row[2])]
    r_score = row[7]
    red = red.copy()
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
    # get the average of things we want
    avg_red = red.apply(np.mean, axis = 0)
    r_rank = avg_red["rank"]
    r_OPR = avg_red["OPR"]
    r_DPR = avg_red["DPR"]
    r_CCWM = avg_red["CCWM"]
    # get level of competition (qualification vs elimination)
    level = row["level"]
    
    return pd.Series([b_score, r_score, b_experience, b_wins, b_losses, b_ties, b_points, b_auto, b_rotor, b_touchpad, b_pressure, b_rank, b_OPR, b_DPR, b_CCWM, r_experience, r_wins, r_losses, r_ties, r_points, r_auto, r_rotor, r_touchpad, r_pressure, r_rank, r_OPR, r_DPR, r_CCWM, level])

def calculateAlliance(matches, teams):
    """ 
        @param matches: dataframe with match information
        @param teams: dataframe with team statistics
        @return df: dataframe with alliance statistics that are summed or averaged of the individual teams
    """
    df = matches.apply(lambda row: alliance(row, teams), axis = 1)
    df.columns = ["b_score", "r_score", "b_experience", "b_wins", "b_losses", "b_ties", "b_points", "b_auto", "b_rotor", "b_touchpad", "b_pressure", "b_rank", "b_OPR", "b_DPR", "b_CCWM", "r_experience", "r_wins", "r_losses", "r_ties", "r_points", "r_auto", "r_rotor", "r_touchpad", "r_pressure", "r_rank", "r_OPR", "r_DPR", "r_CCWM", "level"]
    return df

# Get the offensive player rating (OPR)
# https://erikrood.com/Python_References/rows_cols_python.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html
# http://book.pythontips.com/en/latest/enumerate.html
# # https://stackoverflow.com/questions/22963263/creating-a-zero-filled-pandas-data-frame
def OPR(teams, qm):
    """ 
        @param teams: list of team names
        @param qm: dataframe with qualification match information
        @return OPR: list with OPR scores for each team
    """
    teams.append("score")
    m = np.zeros((len(teams), len(teams)))
    for counter, team in enumerate(teams):
        # create dataframe of only games played by one team
        blue_team = qm.loc[(qm["b1"] == team) | (qm["b2"] == team) | (qm["b3"] == team)]
        red_team = qm.loc[(qm["r1"] == team) | (qm["r2"] == team) | (qm["r3"] == team)]
        indiv_team = pd.concat([blue_team, red_team])
        indiv_team.reset_index(inplace = True)
        matrix = pd.DataFrame(0, index = np.arange(len(indiv_team)), columns = teams)
        # update dataframe with scores and who played in the match
        for index, row in indiv_team.iterrows():
            b1 = row["b1"]
            b2 = row["b2"]
            b3 = row["b3"]
            blue = []
            blue.extend([b1, b2, b3])
            bscore = row["bscore"]
            r1 = row["r1"]
            r2 = row["r2"]
            r3 = row["r3"]
            red = []
            red.extend([r1, r2, r3])
            rscore = row["rscore"]
            if team in blue:
                matrix.loc[index, b1] = 1
                matrix.loc[index, b2] = 1
                matrix.loc[index, b3] = 1
                matrix.loc[index, "score"] = bscore
            else:
                matrix.loc[index, r1] = 1
                matrix.loc[index, r2] = 1
                matrix.loc[index, r3] = 1
                matrix.loc[index, "score"] = rscore
        new_row = matrix.sum(axis = 0)
        m[counter, :] = new_row
    M = m.copy()
    M = M[:, 0:len(teams)-1]
    s = m.copy()
    s = s[:, -1]
    OPR = np.linalg.lstsq(M,s)[0]
    return OPR

# Get the defensive player rating (DPR)
def DPR(teams, qm):
    """ 
        @param teams: list of team names
        @param qm: dataframe with qualification match information
        @return DPR: list with DPR scores for each team
    """
    m = np.zeros((len(teams), len(teams)))
    for counter, team in enumerate(teams):
        # create dataframe of only games played by one team
        blue_team = qm.loc[(qm["b1"] == team) | (qm["b2"] == team) | (qm["b3"] == team)]
        red_team = qm.loc[(qm["r1"] == team) | (qm["r2"] == team) | (qm["r3"] == team)]
        indiv_team = pd.concat([blue_team, red_team])
        indiv_team.reset_index(inplace = True)
        matrix = pd.DataFrame(0, index = np.arange(len(indiv_team)), columns = teams)
        # update dataframe with scores and who played in the match
        for index, row in indiv_team.iterrows():
            b1 = row["b1"]
            b2 = row["b2"]
            b3 = row["b3"]
            blue = []
            blue.extend([b1, b2, b3])
            bscore = row["bscore"]
            r1 = row["r1"]
            r2 = row["r2"]
            r3 = row["r3"]
            red = []
            red.extend([r1, r2, r3])
            rscore = row["rscore"]
            if team in blue:
                matrix.loc[index, b1] = 1
                matrix.loc[index, b2] = 1
                matrix.loc[index, b3] = 1
                matrix.loc[index, "score"] = rscore
            else:
                matrix.loc[index, r1] = 1
                matrix.loc[index, r2] = 1
                matrix.loc[index, r3] = 1
                matrix.loc[index, "score"] = bscore
        new_row = matrix.sum(axis = 0)
        m[counter, :] = new_row
    M = m.copy()
    M = M[:, 0:len(teams)-1]
    s = m.copy()
    s = s[:, -1]
    DPR = np.linalg.lstsq(M,s)[0]
    return DPR

def CCWM(row):
    """
    @param row: row from apply fn
    @return: CCWM for teams
    """
    return row["OPR"] - row["DPR"]

# Function to get OPR, DPR, and Calculated Contribution to Winning Margin (CCWM) and put it in one dataframe
def get_OPR(teams, qm):
    """ 
        @param teams: list of team names
        @param qm: dataframe with qualification match information
        @return df_OPR: dataframe with OPR, DPR, and CCWM for each team
    
    """
    opr = OPR(teams, qm)
    dpr = DPR(teams, qm)
    df_OPR = pd.DataFrame(columns = ["team", "OPR", "DPR", "CCWM"])
    for counter, team in enumerate(teams):
        if team != "score":
            df_OPR = df_OPR.append({"team": team, "OPR": opr[counter], "DPR": dpr[counter]}, ignore_index = True)
    df_OPR["CCWM"] = df_OPR.apply(CCWM, axis = 1)
    return df_OPR


# Generate a df with all of the features and targets we want
path = 'C:/Users/Ithier/Documents/CSCI 29/Grad Project/FIRST/Data/'
event_keys_toload = []
for event in event_keys:
    try:
        experience = get_experience(event, headers)
        stats = get_stats(event, headers)
        matches = get_matches(event, headers)
        # get quarterfinal matches
        qm = matches.loc[matches["level"] == "qm"]
    # https://stackoverflow.com/questions/1966207/converting-numpy-array-into-python-list-structure
        teams = experience["team"].tolist()
        opr = get_OPR(teams, qm)
        my_dfs = [experience, stats, opr]
        total_stats = reduce(lambda x,y: pd.merge(x,y, on = "team"), my_dfs)
        df = calculateAlliance(matches, total_stats)
        name = path + event + '.csv'
        df.to_csv(name)
        event_keys_toload.append(event)
    except:
        print("No data for " + event)