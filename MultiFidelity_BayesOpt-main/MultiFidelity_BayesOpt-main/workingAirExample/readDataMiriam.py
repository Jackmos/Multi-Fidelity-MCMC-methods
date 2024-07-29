# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:40:43 2024

@author: Lips
"""

#!pip install pandas
#!pip install openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from scipy.signal import butter, filtfilt, lfilter
#%% define some filters 
def low_pass_filter(df, cutoff_frequency, target_sampling_timeDelta, order=5, use_filtfilt=False):
    # Resample the data to a regular interval
    df = df.dropna()
    df_resampled = df.resample(target_sampling_timeDelta).mean().interpolate(method='linear')
    df_resampled = df;
    # Apply low-pass filter to the resampled data
    fs = 1 / target_sampling_timeDelta.total_seconds()
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if use_filtfilt:
        filtered_data = filtfilt(b, a, df_resampled)
    else:
        filtered_data = lfilter(b, a, df_resampled)

    # Create a new DataFrame with the filtered column
    df_filtered = pd.DataFrame(filtered_data, index=df_resampled.index)

    return df_filtered

#%%
dataPath = r"C:/Users/Lips/Documents/combineBayesOptAndMF/Daten Miriam/"
start_datetime = pd.to_datetime('2020-05-28 15:00:00')
end_datetime = pd.to_datetime('2020-06-06')

# resampling and symmetric data filtering for high-res LF data
target_sampling_timeDelta = pd.Timedelta(seconds=2)
cutoff_freq = 1/120


# use data from 2020.5.28 to 2020.6.05

# get high res data from Grimm
dataName = "Grimm EDM 28.5 - 07.06_1 min.xlsx"
df = pd.read_excel(dataPath + dataName,parse_dates=['Datum/Zeit'], index_col='Datum/Zeit')
df.rename(columns=lambda x: x.replace("Römerschule ", "grimm_"), inplace=True)
for col in df.columns:
    if 'grimm_' not in col:
        df.rename(columns={col: 'grimm_' + col}, inplace=True)
df = df.rename_axis('timestamp')
df.dropna(axis=1, how='all', inplace=True)
#drop all bins
phrase_to_drop = 'µm'
columns_to_drop = [col for col in df.columns if phrase_to_drop in col]
df = df.drop(columns=columns_to_drop)
df.index = pd.to_datetime(df.index)
df.index = df.index + pd.Timedelta(hours=1+6/60)
grimm_index = df.index
dfAll = df

# get low res data from boxes
# first get the headers
dataName = "Headings Boxes Sensoren.xlsx"
dfheaders = pd.read_excel(dataPath + dataName, header = 0)
headers = dfheaders.keys().tolist()

# get the box data
# resampling time
first_highres_datetime = pd.to_datetime('2020-05-28 15:00:00')

dataName = "2020.5.28 - 2020.6.05 B03AS04M03.TXT"
df = pd.read_csv(dataPath + dataName, sep='\t', header=None, names=headers, parse_dates=['Date Time'], index_col='Date Time')
df = df.rename_axis('timestamp')
df = df.add_prefix('B03_')
df.dropna(axis=1, how='all', inplace=True)
#drop all bins
phrase_to_drop = 'Bin'
columns_to_drop = [col for col in df.columns if ((phrase_to_drop in col) and not ('Bin0' in col))]
df = df.drop(columns=columns_to_drop)
#resample to 2s starting from origin point
df = df.resample(target_sampling_timeDelta, origin = first_highres_datetime).mean().interpolate(method='linear')
for col in df.keys():
    df[col] = low_pass_filter(df[col], cutoff_freq, target_sampling_timeDelta, use_filtfilt=True)
df.index = pd.to_datetime(df.index)
df.index = df.index + pd.Timedelta(seconds=26)
dfAll = pd.merge(dfAll, df, left_index=True, right_index=True, how='outer')


dataName = "2020.5.28 - 2020.6.05 B05AS03M08  no dryer.TXT"
df = pd.read_csv(dataPath + dataName, sep='\t', header=None, names=headers, parse_dates=['Date Time'], index_col='Date Time')
df = df.rename_axis('timestamp')
df = df.add_prefix('B05_')
df.dropna(axis=1, how='all', inplace=True)
#drop all bins
phrase_to_drop = 'Bin'
columns_to_drop = [col for col in df.columns if ((phrase_to_drop in col) and not ('Bin0' in col))]
df = df.drop(columns=columns_to_drop)
#resample to 2s starting from origin point
df = df.resample(target_sampling_timeDelta, origin = first_highres_datetime).mean().interpolate(method='linear')
for col in df.keys():
    df[col] = low_pass_filter(df[col], cutoff_freq, target_sampling_timeDelta, use_filtfilt=True)
df.index = pd.to_datetime(df.index)
df.index = df.index + pd.Timedelta(seconds=37)
dfAll = pd.merge(dfAll, df, left_index=True, right_index=True, how='outer')

dataName = "2020.5.28 - 2020.6.05 B06AS09M10.TXT"
df = pd.read_csv(dataPath + dataName, sep='\t', header=None, names=headers, parse_dates=['Date Time'], index_col='Date Time')
df = df.rename_axis('timestamp')
df = df.add_prefix('B06_')
df.dropna(axis=1, how='all', inplace=True)
#drop all bins
phrase_to_drop = 'Bin'
columns_to_drop = [col for col in df.columns if ((phrase_to_drop in col) and not ('Bin0' in col))]
df = df.drop(columns=columns_to_drop)
#resample to 2s starting from origin point
df = df.resample(target_sampling_timeDelta, origin = first_highres_datetime).mean().interpolate(method='linear')
for col in df.keys():
    df[col] = low_pass_filter(df[col], cutoff_freq, target_sampling_timeDelta, use_filtfilt=True)
df.index = pd.to_datetime(df.index)
df.index = df.index + pd.Timedelta(seconds=10)
dfAll = pd.merge(dfAll, df, left_index=True, right_index=True, how='outer')

dataName = "SDS011 Jan - July.TXT"
df = pd.read_csv(dataPath + dataName, sep='\t', header=0, names=['timestamp', 'PM10', 'PM2.5'])
df = df[df['timestamp'].str.count('\.') == 2] \
                                 [df['timestamp'].str.count(':') == 2]
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%Y %H:%M:%S')
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

df.set_index('timestamp', inplace=True)
df = df.add_prefix('SDS011_')
df.dropna(axis=1, how='all', inplace=True)
df.index = pd.to_datetime(df.index)
df.index = df.index + pd.Timedelta(hours=9/60+12/60/60)
df = df.loc[start_datetime:end_datetime]
dfAll.index = pd.to_datetime(dfAll.index)
dfAll = pd.merge(dfAll, df, left_index=True, right_index=True, how='outer')

# drop all empty columns
dfAll.dropna(axis=1, how='all', inplace=True)
dfAll.index = pd.to_datetime(dfAll.index)
dfAll = dfAll.loc[start_datetime:end_datetime]
dfAll.index = pd.to_datetime(dfAll.index)

# get training data: only keep grimm points and interpolate others to match Grimm times
dfTrain = dfAll.interpolate()
dfTrain = dfTrain.reindex(grimm_index)
dfTrain.to_csv('dfTrain_AirQuality.csv')

#%%
plt.figure()
plt.title('PM10')
for key in ['B03_PM10', 'B05_PM10', 'B06_PM10', 'grimm_PM10', 'SDS011_PM10']:
    plt.plot(dfTrain[key].dropna(), label = key)

plt.figure()
plt.title('PM2.5')
for key in ['B03_PM2.5', 'B05_PM2.5', 'B06_PM2.5', 'grimm_PM2.5', 'SDS011_PM2.5']:
    plt.plot(dfAll[key].dropna(), label = key)
    
plt.figure()
plt.title('PM1')
for key in ['B03_PM1', 'B05_PM1', 'B06_PM1', 'grimm_PM1']:
    plt.plot(dfAll[key].dropna(), label = key)

plt.figure()
plt.title('Relative Humidity')
for key in ['B03_RH OPC', 'B05_RH OPC', 'B06_RH OPC', 'B03_RH HYT', 'B05_RH HYT', 'B06_RH HYT', 'grimm_Feuchte']:
    plt.plot(dfAll[key].dropna(), label = key)
    
plt.figure()
plt.title('Temperature')
for key in ['B03_Temp OPC', 'B05_Temp OPC', 'B06_Temp OPC', 'B03_Temp HYT', 'B05_Temp HYT', 'B06_Temp HYT', 'grimm_Temperatur']:
    plt.plot(dfAll[key].dropna(), label = key)
    
