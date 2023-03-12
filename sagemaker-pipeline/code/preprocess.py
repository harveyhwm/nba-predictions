import os, re
import glob
import boto3
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

import subprocess

from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder

from collections import Counter, deque, defaultdict
import datetime
from datetime import timedelta as td
from datetime import datetime as dt

from zipfile import ZipFile
import sys

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket = 'sagemaker-pipelines-hwm'

input_path = '/opt/ml/processing/input'
train_path = '/opt/ml/processing/train'
val_path = '/opt/ml/processing/val'
test_path = '/opt/ml/processing/test'

def read_s3(file_path, bucket='hwm-nba', output=None, columns=None):
    data = s3_resource.Object(bucket, file_path).get()['Body'].read()
    if file_path[-3:] == 'zip':
        data = zipfile.ZipFile(BytesIO(data))
        data = data.read(data.namelist()[0])
    data = data.decode('utf-8')
    if output == 'dataframe':
        kwargs = {'header': 0}
        if columns is not None: kwargs['names'] = columns
        data = pd.read_csv(StringIO(data), **kwargs)
    return data


def basic_features(games):
    filters = []
    filters.append("away != 'No Games'")
    filters.append("away != '0'")
    filters.append("hs >= 50")
    filters.append("hs != '-'")
    filters.append("diff != 0")
    filters.append("date >= '1996'")

    # games.columns = [re.sub(' ','_',c.lower()) for c in list(games.columns)]
    games = games.drop_duplicates(subset=['detail_path'], keep='last')
    
    games.loc[:,'hs'] = games.loc[:,'hs'].apply(lambda x: make_int(x))
    games.loc[:,'as'] = games.loc[:,'as'].apply(lambda x: make_int(x))

    # clean up home and away scores
    games['ot'] = games.apply(lambda x: 0 if (x['ot']=='N' or x['away']=='No Games') \
                              else (1 if int(max(x['as'],x['hs']))<200 else int(str(max(x['as'],x['hs']))[:1])),axis=1)

    games['as'] = games.apply(lambda x: x['as'] if (x['ot']<2 or x['as']<200) else x['as'] % 1000, axis=1)
    games['hs'] = games.apply(lambda x: x['hs'] if (x['ot']<2 or x['hs']<200) else x['hs'] % 1000, axis=1)

    games['game_subtype'] = games['game_type'].apply(lambda x: re.sub('^(east|west)-', '', re.sub('-[0-9]{1}$','',x)))
    games['game_type'] = games['game_type'].apply(lambda x: x if x in ['preseason','regular'] else 'playoffs')

    games['diff'] = games.apply(lambda r: np.abs(r['hs']-r['as']),axis=1)
    games = games.query(' & '.join(filters)).reset_index(drop=True)
    
    # clean the team names - reconcile old franchises with new, and aggregate the long tail of guest/novelty teams
    games['home'] = games['home'].apply(lambda x: team_names_dict[x])
    games['away'] = games['away'].apply(lambda x: team_names_dict[x])
        
    games['month'] = games['date'].apply(lambda x: x[5:7])
    games['date'] = games['date'].apply(lambda x: dt.strptime(x,'%Y-%m-%d').date())
    games['winner'] = games.apply(lambda x: x['home'] if x['hs'] >= x['as'] else x['away'], axis=1)
    games['home_win'] = games.apply(lambda x: 1 if x['hs'] >= x['as'] else 0, axis=1)
    games['season'] = games['date'].apply(lambda x: str((x-td(days=225)).year)+'-'+str((x-td(days=225)).year+1)) # mid-august split
    # exceptions
    games['season'] = games.apply(lambda x: '2019-2020' if (dt.strftime(x['date'],'%Y-%m-%d')>='2020-03-01' \
                                                            and dt.strftime(x['date'],'%Y-%m-%d')<='2020-11-15') else x['season'], axis=1)

    games['is_preseason'] = games.apply(lambda x: 1 if x['game_type']=='preseason' else 0,axis=1)
    games['is_playoffs'] = games.apply(lambda x: 0 if x['game_type'] in ['preseason','regular'] else 1,axis=1)
    
    games['team_pair'] = games.apply(lambda r: [r['away'], r['home']], axis=1)
    games['team_pair'].apply(lambda r: r.sort())
    games['team_pair'] = games['team_pair'].apply(lambda r: '-'.join(r))
    
    del games['detail_data']
    del games['rand']
    
    return games


def playoff_features(games): # etl the playoff calculations
    games.columns = [re.sub(' ','_',c.lower()) for c in list(games.columns)]
    playoff_games = games[games['game_type']=='playoffs'].iloc[:,:].sort_values(by=['season', 'team_pair', 'date']).reset_index(drop=True)

    playoff_games['winner_1'] = playoff_games.apply(lambda r: 1 if r['winner']==r['team_pair'][:3] else 0, axis=1)
    playoff_games['winner_2'] = playoff_games.apply(lambda r: 1 if r['winner']==r['team_pair'][-3:] else 0, axis=1)

    playoff_games[['t1_wins_after_game', 't2_wins_after_game']] = \
        playoff_games[['season', 'team_pair', 'winner_1', 'winner_2']].groupby(['season', 'team_pair'])[['winner_1', 'winner_2']].transform(pd.Series.cumsum)
    del playoff_games['winner_1']
    del playoff_games['winner_2']

    playoff_games['leader_after_game'] = playoff_games.apply(lambda r: r['team_pair'][:3] if r['t1_wins_after_game']>r['t2_wins_after_game'] \
                                                             else (r['team_pair'][-3:] if r['t2_wins_after_game']>r['t1_wins_after_game'] \
                                                                   else 'tied series'), axis=1)
    playoff_games[['season2', 'team_pair2', 't1_wins_before_game', 't2_wins_before_game', 
                   'leader_before_game']] = playoff_games[['season', 'team_pair', 't1_wins_after_game',
                                                           't2_wins_after_game', 'leader_after_game']].shift(periods=1)
    
    playoff_games['t1_wins_before_game'] = playoff_games.apply(lambda r: 0 if np.isnan(r['t1_wins_before_game']) \
                                                               or r['season']!=r['season2'] else int(r['t1_wins_before_game']), axis=1)
    playoff_games['t2_wins_before_game'] = playoff_games.apply(lambda r: 0 if np.isnan(r['t2_wins_before_game']) \
                                                               or r['season']!=r['season2'] else int(r['t2_wins_before_game']), axis=1)
    playoff_games['leader_before_game'] = playoff_games.apply(lambda r: 'series starting' \
                                                              if str(r['leader_before_game'])=='nan' or r['season']!=r['season2'] \
                                                              else r['leader_before_game'], axis=1)
    del playoff_games['season2']
    del playoff_games['team_pair2']
    
    playoff_series_winners = playoff_games.drop_duplicates(subset=['season', 'team_pair'],keep='last').reset_index(drop=True)
    playoff_series_winners = playoff_series_winners[['date', 'season', 'game_type', 'team_pair', 'winner']]
    playoff_series_winners = playoff_series_winners.sort_values(by=['winner', 'date']).reset_index(drop=True)
    playoff_series_winners['count'] = 1

    playoff_series_winners['playoff_round'] = playoff_series_winners[['winner', 'season', 'game_type',
                                                                      'count']].groupby(['winner', 'season', 'game_type'])['count'].transform(pd.Series.cumsum)
    del playoff_series_winners['count']

    playoff_series_winners = playoff_series_winners[['winner', 'season', 'game_type', 'team_pair', 'playoff_round']]
    playoff_series_winners.columns = ['series_winner', 'season', 'game_type', 'team_pair', 'playoff_round']

    playoff_games = playoff_games.merge(playoff_series_winners, how='left', on=['season', 'team_pair', 'game_type'], sort=False)
    
    playoff_games['knockout_game'] = playoff_games.apply(lambda r: 1 if max(r['t1_wins_before_game'], r['t2_wins_before_game'])==3 \
                                                         else (1 if max(r['t1_wins_before_game'], r['t2_wins_before_game'])==2 and \
                                                               r['playoff_round']==1 and r['season'] < '2002' else 0), axis=1)
    
    playoff_games = playoff_games[['date', 'detail_path', 't1_wins_after_game', 't2_wins_after_game', 'leader_after_game', 't1_wins_before_game',
                                   't2_wins_before_game', 'leader_before_game', 'series_winner', 'playoff_round', 'knockout_game']]
    
    games = games.merge(playoff_games, how='left', on=['date', 'detail_path'], sort=False)
    
    games['game_subtype'] = games['playoff_round'].apply(lambda x: 0 if np.isnan(x) else int(x))
    del games['playoff_round']

    return games


def wins_n_games(games):
    teams_unique = list(set(get_team_names(games, output='df')['team']))+['Other']
    sequences = [3, 5, 10, 20, 50, 100]
    for j, team in enumerate(teams_unique):
        temp = games[games['team_pair'].str.contains(team)][['detail_path','home','away','team_pair','winner']]
        temp_home = games[games['home'].str.contains(team)][['detail_path','home','winner']]
        temp_away = games[games['away'].str.contains(team)][['detail_path','away','winner']]
        
        wins = np.array(np.where(temp['winner']==team, 1, 0))
        wins_home = np.array(np.where(temp_home['winner']==team, 1, 0))
        wins_away = np.array(np.where(temp_away['winner']==team, 1, 0))
        
        # streak test
        streak = [0,wins[0]]
        for i in range(1, len(wins)-1):
            next_val = streak[-1]+1 if wins[i]==1 else 0
            streak.append(next_val)
        temp['streak'] = np.asarray(streak)
        temp['streak_home'] = np.where(temp['home']==team, temp['streak'], -1)
        temp['streak_away'] = np.where(temp['away']==team, temp['streak'], -1)

        # streak home test
        streak_home = [0,wins_home[0]]
        for i in range(1, len(wins_home)-1):
            next_val = streak_home[-1]+1 if wins_home[i]==1 else 0
            streak_home.append(next_val)
        temp_home['streak_home_home'] = np.asarray(streak_home)
        
        # streak away test
        streak_away = [0,wins_away[0]]
        for i in range(1, len(wins_away)-1):
            next_val = streak_away[-1]+1 if wins_away[i]==1 else 0
            streak_away.append(next_val)
        temp_away['streak_away_away'] = np.asarray(streak_away)
        
        temp = temp.merge(temp_home[['detail_path','streak_home_home']], how='left', on='detail_path', sort=False)
        temp = temp.merge(temp_away[['detail_path','streak_away_away']], how='left', on='detail_path', sort=False)
        
        for s in sequences:
            temp['wins'+str(s)] = np.asarray([sum(wins[max(0,i-s):i]) for i in range(len(wins))])
            temp['wins'+str(s)+'_home'] = np.where(temp['home']==team, temp['wins'+str(s)], -1)
            temp['wins'+str(s)+'_away'] = np.where(temp['away']==team, temp['wins'+str(s)], -1)
        wins_df = temp if j==0 else pd.concat([wins_df, temp], axis=0)

        # return wins_df
    
    wins_df = wins_df[~wins_df['team_pair'].str.contains('Other')]

    home_streaks = wins_df[wins_df['streak_home']>-1].sort_index()
    away_streaks = wins_df[wins_df['streak_away']>-1].sort_index()

    home_wins = wins_df[wins_df['wins3_home']>-1].sort_index()
    away_wins = wins_df[wins_df['wins3_away']>-1].sort_index()
    
    wins_df = home_wins[['detail_path']]
    wins_df = wins_df.merge(home_streaks[['detail_path', 'streak_home', 'streak_home_home']], how='left', on='detail_path', sort=False)
    wins_df = wins_df.merge(away_streaks[['detail_path', 'streak_away', 'streak_away_away']], how='left', on='detail_path', sort=False)
    wins_df = wins_df.merge(home_wins[['detail_path']+['wins'+str(s)+'_home' for s in sequences]], how='left', on='detail_path', sort=False)
    wins_df = wins_df.merge(away_wins[['detail_path']+['wins'+str(s)+'_away' for s in sequences]], how='left', on='detail_path', sort=False)
    for col in list(wins_df.columns)[1:]:
        wins_df[col] = wins_df[col].fillna(0).astype(int)

    games = games.merge(wins_df, how='inner', on=['detail_path'], sort=-False).reset_index(drop=True)
    return games


def opponents(games):
    games['team_pair_sorted'] = np.asarray([g[:3]+'-'+g[-3:] if g[:3]<=g[-3:] else g[-3:]+'-'+g[:3] for g in games['team_pair']])
    unique_pairs = np.unique(games['team_pair_sorted'])

    for j, teams in enumerate(unique_pairs):
        temp = games[games['team_pair_sorted']==teams][['detail_path','team_pair_sorted','winner']]
        wins = np.array(np.where(temp['winner']==teams[:3], 1, 0))

        streak1, streak2 = [0], [0]
        for i in range(len(wins)-1):
            next_val1 = streak1[-1]+1 if wins[i]==1 else 0
            streak1.append(next_val1)
            next_val2 = streak2[-1]+1 if wins[i]==0 else 0
            streak2.append(next_val2)
        temp['streak1'] = np.asarray(streak1)
        temp['streak2'] = np.asarray(streak2)

        opponents_df = temp if j==0 else pd.concat([opponents_df, temp], axis=0)

    games = games.merge(opponents_df[['detail_path','team_pair_sorted','streak1','streak2']],
                        how='left', on=['detail_path', 'team_pair_sorted'], sort=False)
    games['streak_opponents_home'] = games.apply(lambda r: r['streak1'] if r['home'] == r['team_pair_sorted'][:3] else r['streak2'], axis=1)
    games['streak_opponents_away'] = games.apply(lambda r: r['streak1'] if r['away'] == r['team_pair_sorted'][:3] else r['streak2'], axis=1)

    del games['streak1']
    del games['streak2']

    return games


def last_n_days(games):
    teams_unique = list(set(get_team_names(games, output='df')['team'])) # +['Other']
    sequences = [2, 3, 5, 10, 20]

    min_date = min(games['date'])
    max_date = max(games['date'])
    dates = [min_date + datetime.timedelta(days=d) for d in range((max_date - min_date).days)]

    for j, team in enumerate(teams_unique):
        temp = games[games['team_pair'].str.contains(team)][['date','detail_path','home','away']]
        d = np.array(temp['date'])
        temp['days_last_played'] = [0]+[(d[i]-d[i-1]).days for i in range(1,len(d))]

        y_dates = np.array(temp['date'])
        y_df = pd.DataFrame(y_dates,columns=['date'])
        y_df['game'] = 1
        n_dates = list(set(dates) - set(y_dates))
        n_df = pd.DataFrame(n_dates,columns=['date'])
        n_df['game'] = 0
        df = pd.concat([y_df, n_df], axis=0).sort_values(by='date').reset_index(drop=True)

        for s in sequences:
            game_seq = np.array(df['game'])
            cum_games = np.asarray([0]+[sum(game_seq[max(0,i-s):i]) for i in range(1,len(game_seq))])
            df['cum_games_'+str(s)] = cum_games

        temp_home = temp[temp['home']==team]
        temp_home = temp_home.merge(df, how='inner', on=['date'], sort=False)
        temp_home_all = temp_home if j==0 else pd.concat([temp_home_all, temp_home], axis=0)

        temp_away = temp[temp['away']==team]
        temp_away = temp_away.merge(df, how='inner', on=['date'], sort=False)
        temp_away_all = temp_away if j==0 else pd.concat([temp_away_all, temp_away], axis=0)

    temp_home_all = temp_home_all[['date','detail_path']+['cum_games_'+str(s) for s in sequences]+['days_last_played']]
    temp_home_all.columns = ['date','detail_path']+[c+'_home' for c in list(temp_home_all.columns)[2:]]

    temp_away_all = temp_away_all[['date','detail_path']+['cum_games_'+str(s) for s in sequences]+['days_last_played']]
    temp_away_all.columns = ['date','detail_path']+[c+'_away' for c in list(temp_away_all.columns)[2:]]        

    games = games.merge(temp_home_all, how='left', on=['date','detail_path'], sort=False)
    games = games.merge(temp_away_all, how='left', on=['date','detail_path'], sort=False)

    return games


def generate_standings(table, date, conf=None):
    if type(date) == str:
        date = datetime.datetime.strptime(date,'%Y-%m-%d').date()
    if type(table['date'][0]) == str:
        table['date'] = table['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d').date())

    season = max(table[table['date']<date]['season'])
    games_season = table[(table['season']==season) & (table['date']<date)]
    games_season.loc[:,'count'] = 1
    games_preseason = games_season[games_season['game_type']=='preseason'].reset_index(drop=True)
    games_regular = games_season[games_season['game_type']=='regular'].reset_index(drop=True)
    games_playoffs = games_season[games_season['game_type']=='playoffs'].reset_index(drop=True)

    hw = games_regular[games_regular['home']==games_regular['winner']][['home','count','hs']].groupby('home').sum()
    hl = games_regular[games_regular['home']!=games_regular['winner']][['home','count']].groupby('home').sum()
    aw = games_regular[games_regular['away']==games_regular['winner']][['away','count','as']].groupby('away').sum()
    al = games_regular[games_regular['away']!=games_regular['winner']][['away','count']].groupby('away').sum()

    final_table = pd.concat([hw, hl, aw, al], axis=1).reset_index().fillna(0)

    final_table.columns = ['team', 'wins_home', 'points_home', 'losses_home', 'wins_away', 'points_away', 'losses_away']

    final_table['wins'] = final_table['wins_home']+final_table['wins_away']
    final_table['losses'] = final_table['losses_home']+final_table['losses_away']
    final_table['played'] = final_table['wins']+final_table['losses']
    final_table['diff'] = final_table['wins'] - final_table['losses']
    final_table['conf'] = final_table['team'].map(get_conference)

    final_table = final_table[['team', 'conf', 'played', 'wins', 'losses', 'wins_home', 'losses_home',
                               'points_home', 'wins_away', 'losses_away', 'points_away', 'diff']]
    final_table = final_table.sort_values(by=['diff', 'played'],ascending=[False, True]).reset_index(drop=True)

    if conf is not None:
        final_table = final_table[final_table['conf']==conf].reset_index(drop=True)

    final_table = final_table.reset_index()
    final_table.columns = ['pos']+list(final_table.columns)[1:]
    final_table['pos'] += 1

    return final_table


def build_standings(games):
    for i, d in enumerate(set(games['date'])):
        if i%500==0: print(i)
        try:
            standings = generate_standings(games, d)
            standings['date'] = d
            if i == 0:
                standings_all = standings
            else:
                standings_all = pd.concat([standings_all, standings], axis=0)
        except:
            # print(d)
            pass
    standings_all = standings_all.sort_values(by=['date', 'pos'], ascending=[True, True])
    standings_all.to_csv('../data/processed/standings.csv', index=False)
    return standings_all


def standings(games, standing_path=input_path+'/standings.zip'):
    standing = read_csv_from_zip(standing_path) # build_standings(games)
    try:
        standing['date'] = standing['date'].apply(lambda x: dt.strptime(x,'%Y-%m-%d').date())
    except:
        pass
    standings_home = standing[['team','date','wins','losses','wins_home','losses_home','points_home','wins_away','losses_away','points_away','diff']]
    standings_home.columns = ['team','date']+['home_standing_'+c for c in list(standings_home.columns)[2:]]
    standings_away = standing[['team','date','wins','losses','wins_home','losses_home','points_home','wins_away','losses_away','points_away','diff']]
    standings_away.columns = ['team','date']+['away_standing_'+c for c in list(standings_away.columns)[2:]]

    games = games.merge(standings_home, how='left', left_on=['home','date'], right_on=['team','date'], sort=False)
    games = games.merge(standings_away, how='left', left_on=['away','date'], right_on=['team','date'], sort=False)

    del games['team_x']
    del games['team_y']

    for c in list(games.columns)[-18:]:
        games[c] = games[c].fillna(0)

    games['standing_diff'] = games['home_standing_diff'] - games['away_standing_diff']
    games['standing_points_diff'] = games['home_standing_points_home'] - games['away_standing_points_away']

    return games


def make_int(c):
    try:
        return int(c)
    except:
        return 0


def day_diff(d1,d2,t):
    try:
        return int((d1-d2).days)
    except:
        return np.nan


def delcols(df,c):
    try:
        del df[c]
    except:
        pass


def datefill(d,end=None,numdays=0):
    if end is None:
        end = max(d) #.date()
    dr = (end-min(d)).days #.date()
    dl = [end - datetime.timedelta(days=x) for x in range(dr+1+numdays)]
    # d = [d.date() for d in d]
    date_list = [a for a in dl if a not in d]
    if numdays > 0: return date_list[:numdays]
    return date_list


def get_counts_list(col, name, ascending=False, lim=None):
    data = pd.DataFrame.from_dict([dict(Counter(col))]).T.sort_values(by=0, ascending=ascending).reset_index()
    if lim is not None:
        data = data[:lim]
    data.columns = [name, 'count']
    return data


def get_dupes(col):
    df = pd.DataFrame.from_dict([dict(Counter(col))]).T.sort_values(by=0,ascending=False).reset_index()
    dupes = df[df[0]>1]
    if len(dupes)>0: return dupes
    print('No Dupes')


# team lookup to address the long-tail and reconcile franchises
def get_team_names(games, output='dict', default='Other'):
    team_replacements = {'NOH': 'NOP', 'NOK': 'NOP', 'SAN': 'SAS', 'GOS': 'GSW', 'UTH': 'UTA', 'PHL': 'PHI'}
    active_teams = get_counts_list(games[games['away'] != 'No Games']['home'], 'team_raw', lim=40) # these are the valid teams
    active_teams['team'] = active_teams['team_raw'].apply(lambda x: team_replacements[x] if x in team_replacements.keys() else x)
    if output=='dict':
        def def_value():
            return default
        team_dict = defaultdict(def_value)
        for i in range(len(active_teams)):
            team_dict[active_teams['team_raw'][i]] = active_teams['team'][i]
        return team_dict
    return active_teams[['team_raw', 'team']]


def etl_process(data, tasks):
    for t in tasks:
        data = t(data)
    return data


def get_conference(team=None):
    confs = {'East': ['BOS', 'MIL', 'PHI', 'CLE', 'BKN', 'MIA', 'NYK', 'ATL', 'WAS', 'CHI', 'TOR', 'IND', 'ORL', 'DET', 'CHA'],
             'West': ['DEN', 'MEM', 'SAC', 'PHX', 'DAL', 'LAC', 'NOP', 'MIN', 'GSW', 'OKC', 'UTA', 'POR', 'LAL', 'SAS', 'HOU']}
    if team is not None:
        return 'East' if team in confs['East'] else 'West'
    return confs


def read_csv_from_zip(zip_path):
    zip_file = ZipFile(zip_path)
    file_name = zip_path.split('/')[-1].split('.')[0]+'.csv'
    for z in zip_file.infolist():
        if z.filename == file_name:
            return pd.read_csv(zip_file.open(file_name))

        
def write_s3(file_path, myfile, bucket=bucket, dedupe_cols=None, sort=None, compression='zip'):
    file_name = file_path.split('/')[-1]
    if type(myfile) == pd.core.frame.DataFrame:
        if dedupe_cols is not None:
            myfile = myfile.drop_duplicates(subset=dedupe_cols, keep='first')
        if sort is not None:
            myfile = myfile.sort_values(by=sort, ascending=True)
        output_buffer = BytesIO() if compression == 'zip' else StringIO()
        if compression == 'zip': file_path = '.'.join(file_path.split('.')[:-1])+'.zip'
        myfile.to_csv(output_buffer, index=False, compression={'method':compression, 'archive_name':file_name})
        myfile = output_buffer
    s3_resource.Object(bucket, file_path).put(Body=myfile.getvalue())
        

def encode_features(col):
    le = LabelEncoder()
    le.fit(col)
    col = le.transform(col)
    return col


def standardize_features(col):
    return (col-np.mean(col))/(np.std(col))


def normalize_features(col):
    return (col-min(col))/(max(col)-min(col))


def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package])

        
if __name__ == '__main__':
    
    input_files = os.listdir(input_path)
    
    games = read_csv_from_zip(input_path + '/games.zip')
    games.columns = [re.sub(' ','_',c.lower()) for c in list(games.columns)]
    team_names_dict = get_team_names(games) 
    
    max_played = np.max(np.unique(games[games['as'] != '-']['date']))
    games = games[games['date'] <= max_played].reset_index(drop=True)
    
    games_final = etl_process(games, [basic_features, playoff_features, wins_n_games, opponents, last_n_days, standings])
    
    # making train and test, x and y
    split = int(len(games)*0.8)
    
    target = 'home_win'
    skip_cols = ['date', 'detail_path', 'team_pair', 'team_pair_sorted', 'winner', 'winner', 'hs', 'diff',
                 'as', 'ot', 't1_wins_after_game', 't2_wins_after_game', 'leader_after_game', 'series_winner']
    
    col_map = {}
    for c in games_final.columns:
        if c not in skip_cols: col_map[c] = str(games_final[c].dtype)
    
    for c in col_map.keys():
        if col_map[c] == 'object':
            print('column:',c)
            games_final[c] = games_final[c].astype(str)
            games_final[c] = encode_features(games_final[c])
        
    games_final = games_final.fillna(0)
    games_final = games_final[[c for c in games_final.columns if c not in skip_cols]]
    games_final['standing_diff'] = np.where(games_final['standing_diff']<0, 0, np.where(games_final['standing_diff']<10, 1, 2))
    
    for i, c in enumerate(games_final.columns):
        games_final[c] = standardize_features(games_final[c])
        games_final[c] = normalize_features(games_final[c])
    
    test_frac = 0.1
    train_frac = 1 - test_frac
    
    np.random.seed(56)
    games_final_shuffled = games_final.sample(frac=1).reset_index(drop=True)
    
    train = games_final_shuffled.iloc[:int(len(games_final)*train_frac),:]
    test = games_final_shuffled.iloc[int(len(games_final)*train_frac):,:]
    
    train.to_csv(train_path+'/train.csv', index=False)
    test.to_csv(test_path+'/test.csv', index=False)
    
    s3_client.upload_file(train_path+'/train.csv', bucket, 'nba/processed/train.csv')
    s3_client.upload_file(test_path+'/test.csv', bucket, 'nba/processed/test.csv')
