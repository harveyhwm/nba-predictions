from bs4 import BeautifulSoup as BS
from selenium import webdriver
from io import StringIO
import pandas as pd
import numpy as np
import datetime
import platform
import boto3
import time
import json
import csv
import re

s3_resource = boto3.resource('s3')
lambda_client = boto3.client('lambda')

# helper functions
def make_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--single-process')
    options.add_argument('--disable-dev-shm-usage')
    if platform.system() == 'Linux':
        options.binary_location = '/opt/headless-chromium'
        chromedriver_path = '/opt/chromedriver'
    else:
        chromedriver_path = '/Users/harveymanhood/chromedriver'
    return webdriver.Chrome(chromedriver_path, options=options)

def write_s3(file_path, myfile, bucket='hwm-nba'):
    if type(myfile) == pd.core.frame.DataFrame:
        output_buffer = StringIO()
        myfile.to_csv(output_buffer, index=False)
        myfile = output_buffer
    s3_resource.Object(bucket, file_path).put(Body=myfile.getvalue())

def read_s3(file_path, bucket='hwm-nba', output=None, columns=None):
    data = s3_resource.Object(bucket, file_path).get()['Body'].read().decode('utf-8')
    if output == 'dataframe':
        kwargs = {'header': 0}
        if columns is not None: kwargs['names'] = columns
        data = pd.read_csv(StringIO(data), **kwargs)
    return data

def retrieve_dates(num_dates, hour_offset=30):
    days = int(hour_offset//24)
    hours = hour_offset%24
    max_date = datetime.datetime.today() - datetime.timedelta(days=days, hours=hours)
    dates = [datetime.datetime.strftime(max_date - datetime.timedelta(days=d),'%Y-%m-%d') for d in range(num_dates)]
    return dates

def scrape_pages(**params):
    driver = make_driver()
    site_root = 'https://www.nba.com'
    page_data = []
    if params['table'] == 'games':
        dates = retrieve_dates(params['num_pages']) # take the last 3 dates up to our specified moment
        game_urls = [site_root+'/games?date='+d for d in dates]
        for i, d in enumerate(dates):
            driver.get(game_urls[i])
            page_source = driver.page_source
            file_path = 'scrapes/games_'+d+'.txt'
            if params['upload_scrape'] == 'true': write_s3(file_path, StringIO(page_source))
            page_data.append((page_source,d))
    elif params['table'] in ('boxes','plays'):
        games = read_s3('data/games.csv', output='dataframe') # get current list of games to retrieve box scores from
        game_ids = games['Detail Path'][-params['num_pages']:]
        detail_url = '/box-score#box-score' if params['table'] == 'boxes' else '/play-by-play?period=All'
        urls = [site_root+'/game/'+g+detail_url for g in game_ids]
        for i, g in enumerate(game_ids):
            driver.get(urls[i])
            page_source = driver.page_source
            file_path = 'scrapes/'+params['table']+'_'+g+'.txt'
            if params['upload_scrape'] == 'true': write_s3(file_path, StringIO(page_source))
            page_data.append([page_source,g])
    else:
        pass
    driver.close()
    driver.quit()
    return page_data

def parse_html(page_data, **params):
    if params['table'] == 'games':
        game_data = []
        columns = ['Date','Away','AS','Home','HS','OT','Detail Path','Game Type','Box Data','Play Data']
        for p in page_data:
            page,date = p
            soup = BS(page, 'html.parser') # lxml
            games = soup.select('div[class*="gamecard"]')
            for g in games:
                box_score = g.select('a[data-id*="box-score"], a[data-id*="preview"]')[0]
                if len(box_score) == 0:
                    continue
                game_url = box_score.get_attribute_list('href')[0]
                game = re.split('\\/|\\-',game_url.upper())
                game_id = game_url.split('game/')[1].split('/box-score')[0]
                gametype = 'playoffs' if len(g.select('[class*="gameSeriesText"]'))>0 else 'preseason' if \
                    g.select('[data-is-preseason]')[0].get_attribute_list('data-is-preseason')[0]=='true' else 'regular'
                scores = [c.get_text() for c in g.select('p[class*="MatchupCardScore"]')]
                scores = ['-','-'] if len(scores)<2 else scores
                overtime = 'Y' if g.select('p[class*="GameCardMatchupStatusText"]')[0].get_text()=='FINAL/OT' else 'N'
                data = [date,game[2],scores[0],game[4],scores[-1],overtime,game_id,gametype,0,0]
                game_data.append(data)
            if (games == []) or (game_data == []):
                game_data.append([d,'No Games',0,'No Games',0,'-','-','-',0,0])
        games_new = pd.DataFrame(game_data, columns=columns)
        games_old = read_s3('data/games.csv', output='dataframe')
        games_new = games_old.append(games_new).drop_duplicates(subset=['Date','Detail Path'],keep='last')
        games_new = games_new.sort_values(by=['Date','Detail Path']).reset_index(drop=True)
        write_s3('data/games.csv', games_new)
    elif params['table'] == 'boxes':
        box_data = []
        columns = ['PLAYER','PLAYERINIT','POSITION','MIN','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA',
                    'FT%','OREB','DREB','REB','AST','STL','BLK','TO','PF','PTS','+/-','Detail Path','Home']
        for p in page_data:
            page,detail = p
            soup = BS(page, 'html.parser')
            boxes = soup.select('table[class*="StatsTable_table"]')
            for i, b in enumerate(boxes):
                # inaugurating the column names
                if len(box_data) == 0:
                    box_data.append(['PLAYER','PLAYERINIT','POSITION'])
                    for th in [e.get_text() for e in b.select('thead>tr>th')[1:]]:
                        box_data[-1].append(th)
                    box_data[-1].append('Detail Path')
                tr = b.select('tbody>tr')
                for d in range(len(tr)):
                    if d < len(tr)-1:
                        td = tr[d].select('td')
                        box_data.append([e.get_text() for e in td[0].select('div>a>span:nth-child(2)>span,div>span:nth-child(2)')])
                    else:
                        box_data.append(['TOTALS','',''])
                    if len(box_data[-1]) < 3:
                        for i in range(3-len(box_data[-1])): box_data[-1].append('')
                    for t in [e.get_text() for e in td[1:]]:
                        box_data[-1].append(t)
                    if len(box_data[-1]) < len(box_data[0])-1:
                        for i in range(len(box_data[0])-len(box_data[-1])-1): box_data[-1].append('')
                    box_data[-1].append(detail)
                    box_data[-1].append('Y') # to fix!!! whether its home or away - check
                    # if i%5: time.sleep(10)
        boxes_new = pd.DataFrame(box_data, columns=columns)
        boxes_old = read_s3('data/boxes.csv', output='dataframe')
        boxes_new = boxes_old.append(boxes_new).drop_duplicates(keep='last').reset_index(drop=True)
        write_s3('data/boxes.csv', boxes_new)
    elif params['table'] == 'plays':
        plays_data = []
        columns = ['Minute','Score','Action','Team','Detail Path']
        for p in page_data:
            page,detail = p
            soup = BS(page, 'html.parser')
            plays = soup.select('article[class*="GamePlayByPlayRow_article"]')
            for s in plays:
                clock = s.select('span[class*="GamePlayByPlayRow_clock"]')[0].get_text()
                score = s.select('span[class*="GamePlayByPlayRow_scoring"]')
                score = score[0].get_text() if len(score)>0 else ''
                desc = s.select('span[class*="GamePlayByPlayRow_desc"]')[0].get_text()
                team = 'Home' if s.get_attribute_list('data-is-home-team')[0]=='true' else 'Away'
                plays_data.append([clock,score,desc,team,detail])
        plays_new = pd.DataFrame(plays_data, columns=columns)
        plays_old = read_s3('data/plays.csv', output='dataframe')
        plays_new2 = plays_old.append(plays_new).drop_duplicates(keep='last').reset_index(drop=True)
        write_s3('data/plays.csv', plays_new2)
        # write_s3('data/plays_new.csv', plays_new)
    else:
        pass

# main function
def main(*args): # event, context
    if args:
        params = args[0]
        if 'table' not in params.keys(): params['table'] = 'games'
        if 'num_pages' not in params.keys(): params['num_pages'] = '3'
        if 'upload' not in params.keys(): params['upload_scrape'] = 'false'
        params['num_pages'] = int(params['num_pages'])
        if len(args) > 1: context = args[1]
    else:
        params = {'table':'games', 'num_pages':3}
    page_data = scrape_pages(**params)
    parse_html(page_data, **params)