import pandas as pd

#### Read in inputs
fpath = 'C://Users/srirri02/Documents/Python Scripts/tennis/tennis_slam_pointbypoint/'


# ====== Exploratory analysis using only 2018 wimbledon =========
tourney_name = '2017-wimbledon'
matchfile = fpath + 'data/' + tourney_name + '-matches.csv'
pointfile = fpath + 'data/' + tourney_name + '-points.csv'
match = pd.read_csv(matchfile)
point = pd.read_csv(pointfile)

# ================ Dataset preparation =================

# Convert scores column to numeric, by converting 'AD' to 99 first
point['P1Score'] = pd.to_numeric(point['P1Score'].replace('AD',99))
point['P2Score'] = pd.to_numeric(point['P2Score'].replace('AD',99))

# Tiebreak Indicator
point['Tiebreak'] = (point['P1GamesWon']==6) & (point['P2GamesWon']==6)

# Label the winner of the point on the row of the point itself
point['CurrPointWinner'] = point.groupby('match_id')['PointWinner'].shift(-1)

# === Game Point Column ===
p1_has_gp_5 = (point['PointServer']==1) & (point['P1Score']==40) & ((point['P2Score']!=40) & (point['P2Score']!=99))
p1_has_gp_deuce = (point['PointServer']==1) & (point['P1Score']==99)
p2_has_gp_5 = (point['PointServer']==2) & (point['P2Score']==40) & ((point['P1Score']!=40) & (point['P1Score']!=99))
p2_has_gp_deuce = (point['PointServer']==2) & (point['P2Score']==99)

point['P1GamePoint'] = 0
point['P2GamePoint'] = 0
point.loc[p1_has_gp_5 | p1_has_gp_deuce, 'P1GamePoint'] = 1
point.loc[p2_has_gp_5 | p2_has_gp_deuce, 'P2GamePoint'] = 1

# === Set Point Column ===
# If serving
p1_sp_normal = (point['P1GamePoint']==1) & (point['P1GamesWon']-point['P2GamesWon']>=1) & (point['P1GamesWon']>=5)
p1_sp_tiebreak = (point['Tiebreak']==1) & (point['P1Score']>=6) & (point['P1Score']-point['P2Score']==1)

p2_sp_normal = (point['P2GamePoint']==1) & (point['P2GamesWon']-point['P1GamesWon']>=1) & (point['P2GamesWon']>=5)
p2_sp_tiebreak = (point['Tiebreak']==1) & (point['P2Score']>=6) & (point['P2Score']-point['P1Score']==1)

# If receiving
p1_break_set = (point['P1BreakPoint']==1) & (point['P1GamesWon']-point['P2GamesWon']>=1) & (point['P1GamesWon']>=5)
p2_break_set = (point['P2BreakPoint']==1) & (point['P2GamesWon']-point['P1GamesWon']>=1) & (point['P2GamesWon']>=5)

# Calculate Set Point Columns using above criteria
point['P1SetPoint'] = 0
point['P2SetPoint'] = 0
point.loc[p1_sp_normal | p1_sp_tiebreak | p1_break_set, 'P1SetPoint'] = 1
point.loc[p2_sp_normal | p2_sp_tiebreak | p2_break_set, 'P2SetPoint'] = 1

# === Match Point Column ===
# Need to calculate a match point column, but have to calculate set point columns first


# ================ Dataset preparation =================
def game_score(df):
    """ Returns the game score in the format of "Server Score - Receiver Score """
    df['Score'] = 0
    df.loc[df['PointServer']==0, 'Score'] = '0-0'
    df.loc[df['PointServer']==1, 'Score'] = df['P1Score'].astype(str) + '-' + df['P2Score'].astype(str)
    df.loc[df['PointServer']==2, 'Score'] = df['P2Score'].astype(str) + '-' + df['P1Score'].astype(str)

    return df['Score']


def is_pressure(df, pressure_points):
    """ Determines whether or not a point is a pressure point
    df = the point by point data
    scores = a list of "pressure" points, defined as 'Server - Receiver Score'
    Returns a binary column, 'Pressure' """

    return df['Score'].isin(pressure_points).astype(int)

pressure_points = ['15-40', '30-40', '40-40', '40-99', '30-30', '0-30', '0-40']
point['Score'] = game_score(point)
point['Pressure'] = is_pressure(point, pressure_points=pressure_points)
