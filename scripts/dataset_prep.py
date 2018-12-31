import pandas as pd
import numpy as np

# ================ Dataset preparation =================

def correct_columns(df):
    """ Add in P1Score, P2Score, Tiebreaker, Current Point winner columns
    AD score will be replaced with 99
    df: df with columns replaced and score columns made numeric
    """
    # Convert scores column to numeric, by converting 'AD' to 99 first
    df['P1Score'] = pd.to_numeric(df['P1Score'].replace('AD',99))
    df['P2Score'] = pd.to_numeric(df['P2Score'].replace('AD',99))

    #  Tiebreak Indicator
    df['Tiebreak'] = (df['P1GamesWon']==6) & (df['P2GamesWon']==6)

    # Label the winner of the point on the row of the df itself
    df['CurrWinner'] = df.groupby('match_id')['PointWinner'].shift(-1)

    return df

def create_score_col(df):
    """ Returns the game score in the format of "Server Score - Receiver Score """
    df['Score'] = 0
    df.loc[df['PointServer']==0, 'Score'] = '0-0'
    df.loc[df['PointServer']==1, 'Score'] = df['P1Score'].astype(str) + '-' + df['P2Score'].astype(str)
    df.loc[df['PointServer']==2, 'Score'] = df['P2Score'].astype(str) + '-' + df['P1Score'].astype(str)
    return df

def create_gamepoint_col(df):
    """ Calculates a column of 1/0 to denote whether the serving player has game point
    df (df): dataframe with PointServer, P1Score, and P2Score columns
    """

    # Check columns are available
    necessary_cols = ['PointServer', 'P1Score', 'P2Score']
    assert np.all([c in df.columns for c in necessary_cols]), 'create_gamepoint_col: necessary columns unavailable'

    # boolean conditions to check if P1 has game point (if serving, or P2 has game point (if serving)
    p1_has_gp_normal = (df['PointServer']==1) & (df['P1Score']==40) & ((df['P2Score']!=40) & (df['P2Score']!=99))
    p1_has_gp_deuce = (df['PointServer']==1) & (df['P1Score']==99)
    p2_has_gp_normal = (df['PointServer']==2) & (df['P2Score']==40) & ((df['P1Score']!=40) & (df['P1Score']!=99))
    p2_has_gp_deuce = (df['PointServer']==2) & (df['P2Score']==99)

    df['GamePoint'] = 0
    df.loc[p1_has_gp_normal | p1_has_gp_deuce, 'GamePoint'] = 1
    df.loc[p2_has_gp_normal | p2_has_gp_deuce, 'GamePoint'] = 2

    return df

def create_setpoint_col(df):
    """ Creates a column denoting which player has set point, if any
    df (DataFrame): DataFrame with GamePoint, P1GamesWon, P2GamesWon, Tiebreak, P1BreakPoint, P2BreakPoint columns"""

    # Check columns are available
    necessary_cols = ['GamePoint','P1GamesWon','P2GamesWon','Tiebreak','P1BreakPoint','P2BreakPoint']
    assert np.all([c in df.columns for c in necessary_cols]), 'create_setpoint_col: Necessary columns unavailable'

    # Boolean conditions to check whether setpoint is available
    # If serving
    p1_sp_normal = (df['GamePoint'] == 1) & (df['P1GamesWon'] - df['P2GamesWon'] >= 1) & (df['P1GamesWon'] >= 5)
    p1_sp_tiebreak = (df['Tiebreak'] == 1) & (df['P1Score'] >= 6) & (df['P1Score'] - df['P2Score'] == 1)
    p2_sp_normal = (df['P2GamePoint'] == 1) & (df['P2GamesWon'] - df['P1GamesWon'] >= 1) & (df['P2GamesWon'] >= 5)
    p2_sp_tiebreak = (df['Tiebreak'] == 1) & (df['P2Score'] >= 6) & (df['P2Score'] - df['P1Score'] == 1)

    # If receiving
    p1_break_set = (df['P1BreakPoint'] == 1) & (df['P1GamesWon'] - df['P2GamesWon'] >= 1) & (df['P1GamesWon'] >= 5)
    p2_break_set = (df['P2BreakPoint'] == 1) & (df['P2GamesWon'] - df['P1GamesWon'] >= 1) & (df['P2GamesWon'] >= 5)

    df['SetPoint'] = 0
    df.loc[p1_sp_normal | p1_sp_tiebreak | p1_break_set, 'SetPoint'] = 1
    df.loc[p2_sp_normal | p2_sp_tiebreak | p2_break_set, 'SetPoint'] = 1
    return df

def create_matchpoint_col(df, match_format):
    """ Create a column, denoting 0/1/2 for which player has match point
    df (DataFrame)
    match_format (int): 3 or 5, denoting best of 3 or best of 5 format

    We're going to do this later
    """


def create_pressure_col(df, pressure_points):
    """ Returns a column denoting whether a point was a pressure point
    df = the point by point data
    scores = a list of "pressure" points, defined as 'Server - Receiver Score'
    Returns a binary column, 'Pressure' """
    df['Pressure'] = df['Score'].isin(pressure_points).astype(int)
    return df

# ================ Dataset preparation =================


if __name__ == '__main__':

    #### Read in inputs
    fpath = 'C://Users/srirri02/Documents/Python Scripts/tennis/tennis_slam_pointbypoint/'

    # ====== Exploratory analysis using only 2017 wimbledon =========
    tourney_name = '2017-wimbledon'
    matchfile = fpath + 'data/' + tourney_name + '-matches.csv'
    pointfile = fpath + 'data/' + tourney_name + '-points.csv'
    match = pd.read_csv(matchfile)
    point = pd.read_csv(pointfile)

    merged = point.merge(match[['match_id','year','slam','match_num','player1','player2']],
                         how='left',
                         on='match_id')

    # Adjustable inputs
    pressure_points = ['15-40', '30-40', '40-40', '40-99', '30-30', '0-30', '0-40']

    # Prep Data
    point_adj_col = create_gamepoint_col(create_score_col(correct_columns(merged)))
    pressure = create_pressure_col(point_adj_col, pressure_points)

    # this aggregation won't work because you can't aggregate only by player1
    # people keep switching whether they're player1 or player2 in a given match

    pressure_players_1 = pressure[pressure['match_num'] < 2000] \
        .groupby('player1').agg({'Pressure': 'sum', 'PointNumber': 'count'})

    pressure_players_2 = pressure[pressure['match_num'] < 2000] \
        .groupby('player2').agg({'Pressure': 'sum', 'PointNumber': 'count'})

    p = pressure_players_1 + pressure_players_2
    p.plot(x='PointNumber', y='Pressure', kind='scatter')
    print(p.sort_values('PointNumber', ascending=False).head(10))
    """
    pressure_players = pressure[pressure['match_num']<2000]\
                       .groupby('player1').agg({'Pressure':'sum', 'PointNumber':'count'})\
                       .sort_values('Pressure', ascending=False)

    pressure_players.sort_values('Pressure',ascending=False)
    
    """