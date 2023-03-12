from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.latest_only_operator import LatestOnlyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.hooks.postgres_hook import PostgresHook
from airflow.contrib.sensors.file_sensor import FileSensor
import datetime

#from extras import postgres_hook_batch
#from extras.postgres_hook_batch import PostgresHookBatching

import pandas as pd
import numpy as np
import os,csv,itertools,shutil

from contextlib import closing
from copy import deepcopy

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService

import pickle
import time
from bs4 import BeautifulSoup as BS
import re
import lxml
import soupsieve

import psycopg2.extensions
import psycopg2.extras

from airflow.hooks.dbapi import DbApiHook
from airflow.models.connection import Connection

import os.path
from pathlib import Path

pg_hook = PostgresHook(postgres_conn_id='postgres_alt')
#pg_hook_batch = PostgresHookBatching(postgres_conn_id='postgres_alt')

CHROMEDRIVER_PATH = '/Users/harveymanhood/chromedriver2'
DATA_PATH = '/Users/harveymanhood/Documents/- projects/- portfolio/nba-predictions/data/'
SITE_ROOT = 'https://www.nba.com'

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
service = ChromeService(executable_path=CHROMEDRIVER_PATH)

interval_minutes_scores = 30
interval_minutes_boxes = 30
interval_minutes_plays = 30

with DAG('nba_scores_dag',
    start_date=datetime.datetime.now()+datetime.timedelta(hours=5)-datetime.timedelta(minutes=interval_minutes_scores+1),
    schedule_interval='*/'+str(interval_minutes_scores)+' * * * *' #'00 00 * * *'
) as dag:
    def scrape_data(**kwargs):
        numdays,offset,backfill_mode = kwargs['numdays'],kwargs['offset'],kwargs['backfill_mode']
        if backfill_mode is True:
            try:
                m = pd.read_csv(DATA_PATH+'matches.csv')
                mdr = datetime.datetime.today() - datetime.timedelta(days=1,hours=7)
                dr = (mdr-datetime.datetime.strptime(min(m['Date']),'%Y-%m-%d')).days
                dl = [datetime.datetime.strftime(mdr - datetime.timedelta(days=x),'%Y-%m-%d') for x in range(dr+1+numdays)]
                dd = list(set(m['Date']))
                date_list = [a for a in dl if a not in dd][:numdays]
            except:
                pass
        else:
            date_list = [datetime.datetime.strftime(datetime.datetime.today() - datetime.timedelta(days=x),'%Y-%m-%d') for x in range(offset,offset+numdays)]
        print(date_list)
        matches = []
        score_df = []
        for d in date_list:
            if d != date_list[0]: time.sleep(3)
            driver = webdriver.Chrome(service=service,options=options)
            driver.get(SITE_ROOT+'/games?date='+d)

            # check chromedriver version
            # https://chromedriver.storage.googleapis.com/index.html?path=106.0.5249.61/

            if 'browserVersion' in driver.capabilities:
                print(driver.capabilities['browserVersion'])
            else:
                print(driver.capabilities['version'])

            page_source = driver.page_source
            soup = BS(page_source,'lxml')
            matches = soup.select('div[class*="gamecard"]')
            for m in matches:
                box_score = m.select('a[data-id*="box-score"]')[0]
                if len(box_score) == 0:
                    continue
                game_url = box_score.get_attribute_list('href')[0]
                game = re.split('\\/|\\-',game_url.upper())
                game_id = game_url.split('game/')[1].split('/box-score')[0]
                gametype = 'playoffs' if len(m.select('[class*="gameSeriesText"]'))>0 else 'preseason' if m.select('[data-is-preseason]')[0].get_attribute_list('data-is-preseason')[0]=='true' else 'regular'   
                scores = [c.get_text() for c in m.select('p[class*="MatchupCardScore"]')]
                scores = ['-','-'] if len(scores)<2 else scores
                overtime = 'Y' if m.select('p[class*="GameCardMatchupStatusText"]')[0].get_text()=='FINAL/OT' else 'N'
                data = [d,game[2],scores[0],game[4],scores[-1],overtime,game_id,gametype,0,0]
                score_df.append(data)
            if (matches == []) or (score_df == []):
                score_df.append([d,'No Games',0,'No Games',0,'-','-','-',0,0])
            driver.close()
            driver.quit()
        with open(DATA_PATH+'matches.pickle','wb') as token:
            pickle.dump(score_df,token)

    def transform_data(**kwargs):
        csv_path = DATA_PATH+'matches.csv'
        dims = ['Date','Away','AS','Home','HS','OT','Detail Path','Game Type','Box Data','Play Data']

        with open(DATA_PATH+'matches.pickle','rb') as token:
            matches = pickle.load(token)
        matches = pd.DataFrame(matches,columns=dims)

        #send to csv
        if os.path.isfile(csv_path):
            matches_existing = pd.read_csv(csv_path)
            matches = (matches_existing.append(matches)).sort_values(by=dims[:3],ascending=[True,True,False]).drop_duplicates(subset=dims[:2])  #groupby(dims).sum().reset_index()
        matches.to_csv(csv_path,index=False)

        #send to postgres
    
    def backup_data(**kwargs):
        csv_path = DATA_PATH+'matches.csv'
        csv_path_new = DATA_PATH+'matches_backup.csv'
        if os.path.exists(csv_path):
            shutil.copy(csv_path,csv_path_new)

    latest_only = LatestOnlyOperator(task_id='latest_only')

    extract = PythonOperator(
        task_id='extract',
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=scrape_data,
        op_kwargs={'numdays':5,'offset':1,'backfill_mode':True}
    )

    transform = PythonOperator(
        task_id='transform',
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=transform_data
    )

    backup = PythonOperator(
        task_id='backup',
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=backup_data
    )

    latest_only >> backup
    latest_only >> extract >> transform

with DAG('nba_boxes_dag',
    start_date=datetime.datetime.now()+datetime.timedelta(hours=5)-datetime.timedelta(minutes=interval_minutes_boxes+1),
    schedule_interval='*/'+str(interval_minutes_boxes)+' * * * *' #'00 00 * * *'
) as dag2:
    def scrape_data_box(**kwargs):
        matches = pd.read_csv(DATA_PATH+'matches.csv')
        matches = matches[(matches['Box Data']==0) & (matches['Detail Path']!='-') & (matches['AS']!='-') & (matches['Game Type']!='preseason')].reset_index(drop=True)[-5:] #sample(frac=1)
        tdata = []
        for m in range(len(matches)):
            url = SITE_ROOT+'/game/'+matches.iloc[m,:]['Detail Path']+'/box-score#box-score'
            driver = webdriver.Chrome(service=service,options=options)
            driver.get(url)
            page_source = driver.page_source
            soup = BS(page_source,'lxml')
            boxes = soup.select('table[class*="StatsTable_table"]')
            for b in boxes:
                tdata.append([['PLAYER','PLAYERINIT','POSITION']])
                for th in [e.get_text() for e in b.select('thead>tr>th')[1:]]:
                    tdata[-1][0].append(th)
                tdata[-1][0].append('Detail Path')
                tr = b.select('tbody>tr')
                for d in range(len(tr)):
                    if d < len(tr)-1:
                        td = tr[d].select('td')
                        tdata[-1].append([e.get_text() for e in td[0].select('div>a>span:nth-child(2)>span,div>span:nth-child(2)')])
                    else:
                        tdata[-1].append(['TOTALS','',''])
                    if len(tdata[-1][-1]) < 3:
                        for i in range(3-len(tdata[-1][-1])): tdata[-1][-1].append('')
                    for t in [e.get_text() for e in td[1:]]:
                        tdata[-1][-1].append(t)
                    if len(tdata[-1][-1]) < len(tdata[-1][0])-1:
                        for i in range(len(tdata[-1][0])-len(tdata[-1][-1])-1): tdata[-1][-1].append('')
                    tdata[-1][-1].append(list(matches['Detail Path'])[m])
            driver.close()
            driver.quit()
            time.sleep(0.5)
        with open(DATA_PATH+'boxes.pickle','wb') as token:
            pickle.dump(tdata,token)

    def transform_data_box(**kwargs):
        csv_path = DATA_PATH+'boxes.csv'
        if os.path.isfile(csv_path):
            boxes_df = pd.read_csv(csv_path)
        else:
            boxes_df = None
        with open(DATA_PATH+'boxes.pickle','rb') as token:
            boxes = pickle.load(token)
        for b in range(len(boxes)):
            box = pd.DataFrame(boxes[b])
            box.columns = box.iloc[0,:]
            box['Home'] = 'Y' if b%2==1 else 'N'
            if boxes_df is None:
                boxes_df = box.iloc[1:,:]
            else:
                boxes_df = boxes_df.append(box.iloc[1:,:])
        boxes_df = boxes_df.drop_duplicates()
        boxes_df.to_csv(csv_path,index=False)
        boxes_df['count'] = 1
        boxes_df = boxes_df.loc[:,['Detail Path','count']].drop_duplicates()
        matches = pd.read_csv(DATA_PATH+'matches.csv')
        matches = matches.merge(boxes_df,how='left',on=['Detail Path'],sort=False)
        matches['Box Data'] = matches.apply(lambda x: 1 if x['count'] == 1 else (-1 if x['Box Data'] == 1 else x['Box Data']),axis=1)
        del matches['count']
        matches.to_csv(DATA_PATH+'matches.csv',index=False)

    latest_only = LatestOnlyOperator(task_id='latest_only')

    def backup_data_box(**kwargs):
        csv_path = DATA_PATH+'boxes.csv'
        csv_path_new = DATA_PATH+'boxes_backup.csv'
        if os.path.exists(csv_path):
            if os.path.exists(csv_path_new):
                if os.path.getsize(DATA_PATH+'boxes.csv') >= os.path.getsize(DATA_PATH+'boxes_backup.csv'):
                    shutil.copy(csv_path,csv_path_new)
            else:
                shutil.copy(csv_path,csv_path_new)
            if datetime.datetime.now().minute == 0:
                shutil.copy(csv_path,re.sub('\.csv','_'+str(datetime.datetime.now().microsecond)+'.csv',csv_path_new))

    extract = PythonOperator(
        task_id='extract',
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=scrape_data_box,
        op_kwargs={'num':3}
    )

    transform = PythonOperator(
        task_id='transform',
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=transform_data_box
    )

    backup = PythonOperator(
        task_id='backup',
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=backup_data_box
    )

    latest_only >> backup
    latest_only >> extract >> transform

with DAG('nba_plays_dag',
    start_date=datetime.datetime.now()+datetime.timedelta(hours=5)-datetime.timedelta(minutes=interval_minutes_plays+1),
    schedule_interval='*/'+str(interval_minutes_plays)+' * * * *' #'00 00 * * *'
) as dag3:
    def scrape_data_plays(**kwargs):
        matches = pd.read_csv(DATA_PATH+'matches.csv')
        matches = matches[(matches['Play Data']==0) & (matches['Detail Path']!='-') & (matches['AS']!='-')].reset_index(drop=True)[-5:] #sample(frac=1) #& (matches['Game Type']!='preseason')
        tdata = [['Minute','Score','Action','Team','Detail Path']]
        for m in range(len(matches)):
            detail_path = matches.iloc[m,:]['Detail Path']
            url = SITE_ROOT+'/game/'+detail_path+'/play-by-play?period=All'
            driver = webdriver.Chrome(service=service,options=options)
            driver.get(url)
            page_source = driver.page_source
            soup = BS(page_source,'lxml') #,'html.parser')
            plays = soup.select('article[class*="GamePlayByPlayRow_article"]')
            for p in plays:
                clock = p.select('span[class*="GamePlayByPlayRow_clock"]')[0].get_text()
                score = p.select('span[class*="GamePlayByPlayRow_scoring"]')
                score = score[0].get_text() if len(score)>0 else ''
                desc = p.select('span[class*="GamePlayByPlayRow_desc"]')[0].get_text()
                team = 'Home' if p.get_attribute_list('data-is-home-team')[0]=='true' else 'Away'
                tdata.append([clock,score,desc,team,detail_path])
            driver.close()
            driver.quit()
            time.sleep(0.5)
        with open(DATA_PATH+'plays.pickle','wb') as token:
            pickle.dump(tdata,token)

    def transform_data_plays(**kwargs):
        csv_path = DATA_PATH+'plays.csv'
        if os.path.isfile(csv_path):
            plays_df = pd.read_csv(csv_path)
        else:
            plays_df = None
        with open(DATA_PATH+'plays.pickle','rb') as token:
            plays = pickle.load(token)
        plays = pd.DataFrame(plays)
        plays.columns = plays.iloc[0,:]
        if plays_df is None:
            plays_df = plays.iloc[1:,:]
        else:
            plays_df = plays_df.append(plays.iloc[1:,:])
        plays_df = plays_df.drop_duplicates()
        plays_df.to_csv(csv_path,index=False)
        plays_df['count'] = 1
        plays_df = plays_df.loc[:,['Detail Path','count']].drop_duplicates()
        matches = pd.read_csv(DATA_PATH+'matches.csv')
        matches = matches.merge(plays_df,how='left',on=['Detail Path'],sort=False)
        matches['Play Data'] = matches.apply(lambda x: 1 if x['count'] == 1 else (-1 if x['Play Data'] == 1 else x['Play Data']),axis=1)
        del matches['count']
        matches.to_csv(DATA_PATH+'matches.csv',index=False)

    latest_only = LatestOnlyOperator(task_id='latest_only')

    def backup_data_plays(**kwargs):
        csv_path = DATA_PATH+'plays.csv'
        csv_path_new = DATA_PATH+'plays_backup.csv'
        if os.path.exists(csv_path):
            if os.path.exists(csv_path_new):
                if os.path.getsize(DATA_PATH+'plays.csv') >= os.path.getsize(DATA_PATH+'plays_backup.csv'):
                    shutil.copy(csv_path,csv_path_new)
            else:
                shutil.copy(csv_path,csv_path_new)
            if datetime.datetime.now().minute == 0:
                shutil.copy(csv_path,re.sub('\.csv','_'+str(datetime.datetime.now().microsecond)+'.csv',csv_path_new))

    extract = PythonOperator(
        task_id='extract',
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=scrape_data_plays,
        op_kwargs={'num':3}
    )

    transform = PythonOperator(
        task_id='transform',
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=transform_data_plays
    )

    backup = PythonOperator(
        task_id='backup',
        trigger_rule=TriggerRule.ALL_DONE,
        python_callable=backup_data_plays
    )

    latest_only >> backup
    latest_only >> extract >> transform