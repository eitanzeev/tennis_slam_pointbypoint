import pandas as pd


match = pd.read_csv('tennis_slam_pointbypoint/2011-ausopen-matches.csv')
point = pd.read_csv('tennis_slam_pointbypoint/2011-ausopen-points.csv')

print([c for c in match.columns if c in point])

print(match.head())

