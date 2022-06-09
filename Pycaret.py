##########ライブラリインストール&インポート##########
#ライブラリインポート
from operator import le
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "colab"
import pandas as pd
from pycaret.anomaly import *
import glob
import os
import pyautogui
import plotly.graph_objects as go
import time
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # LogNormをimport
from sklearn.linear_model import LinearRegression
from sklearn.tree import plot_tree
import shap

#ダムタイプ　'gravity' or 'fill'
dam_type = 'gravity'
#model_list = ['abod', 'cluster', 'cof', 'iforest', 'histogram', 'knn', 'lof', 'svm', 'pca', 'mcd', 'sod', 'sos']
model_list = ['iforest']
#学習＋検証＝０、学習＝１、検証＝２
cal_type = 0
#データの時間間隔
dt = 0.01
#特徴量抽出間隔
f_term = 5.12
#特徴量抽出スライド間隔
f_dt = 0.5
##特徴量データ数（学習）（f_dt*(f_nom-1)　がデータの時間以内になるように設定）
f_nom_train = 5
##特徴量データ数（検証）（f_dt*(f_nom-1)　がデータの時間以内になるように設定）
f_nom_test = 81

#変更不要
if dam_type == 'gravity':
    target_axis = 'DAMAXIS'
elif dam_type == 'fill':
    target_axis = 'STREAM'

def main():
    train_file = glob.glob('./#input/train.csv')[0]
    test_file  = glob.glob('./#input/test.csv')[0]
    top_file   = glob.glob('./#input/top.csv')[0]
    bottom_file = glob.glob('./#input/bottom.csv')[0]
    for model in model_list:
        if cal_type == 0:
            pycaret_train(train_file, model)
            pycaret_test(test_file, model)
            rate_fft(top_file, bottom_file)
        elif cal_type == 1:
            pycaret_train(train_file, model)
        elif cal_type == 2:
            pycaret_test(test_file, model)
            rate_fft(top_file, bottom_file)

def rate_fft(file1, file2):
    rate_list = []
    max_list = []
    data_1 = pd.read_csv(file1, header=4)
    data_1.columns = ['timestamp','STREAM','DAMAXIS','U-D']
    data_2 = pd.read_csv(file2, header=4)
    data_2.columns = ['timestamp','STREAM','DAMAXIS','U-D']
    time_list  = np.linspace(0, f_dt*(f_nom_test-1), f_nom_test) 
    if time_list.max() % f_dt != 0:
        raise ValueError("error!")
    for t in range(len(time_list)):
        df_tmp_1 = data_1[(data_1['timestamp'] > time_list[t]) & (data_1['timestamp'] <= time_list[t]+f_term)]
        df_tmp_2 = data_2[(data_2['timestamp'] > time_list[t]) & (data_2['timestamp'] <= time_list[t]+f_term)]
        fft_data_1, freqList_1 = fft(df_tmp_1)
        fft_data_2, freqList_2 = fft(df_tmp_2)
        rate_fft = abs(fft_data_1) / abs(fft_data_2)
        rate_list.append(rate_fft)
        #max_id = rate_fft.tolist().index(rate_fft.max())
        #max_list.append(freqList_1[max_id])
    X, Y = np.meshgrid(time_list, freqList_1)
    Z = np.array(rate_list).T
    plt.figure()
    fig, ax = plt.subplots(nrows=1, figsize=(20,10))
    interval_of_cf = [0.01, 0.1, 1, 10, 100]
    cont =ax.pcolormesh(X, Y, Z, cmap='jet', norm=LogNorm(vmin=10e-2, vmax=10e2))
    #ax.set_ylabel('', fontsize=24)
    #ax.set_yticks([0, 2, 4])
    #ax.plot(time_list, max_list, color="black")
    plt.colorbar(cont)
    plt.savefig('./#output/rate_fft.jpg')
    plt.close

def LinearRe(df):
    mod = LinearRegression(fit_intercept = True)
    lm  = mod.fit(df[["timestamp"]], df[[target_axis]])
    return mod

def fft(df):
    # フーリエ変換
    fft_data = np.fft.fft(df[target_axis])[:int(len(df)/2)]
    freqList = np.fft.fftfreq(len(df), dt)[:int(len(df)/2)]  # 周波数軸
    return fft_data, freqList

def power_spectrum(df):
    fft_data, freqList = fft(df)
    # 振幅スペクトルを計算
    amp = np.abs(fft_data)
    # パワースペクトルの計算（振幅スペクトルの二乗）
    ps = amp**2
    return ps

def entropy(df):
    e = -sum(np.log2(power_spectrum(df)))
    return e

def power_band(df):
    pb = 0
    for i in range(len(fft(df)[1])):
        if fft(df)[1][i] >= 5 and fft(df)[1][i]<=15:
            pb += power_spectrum(df)[i]
    return pb

def mk_feature_val(data, f_nom):
    f_list = []
    time_list  = np.linspace(0, f_dt*(f_nom-1), f_nom) 
    if time_list.max() % f_dt != 0:
        raise ValueError("error!")
    for t in range(len(time_list)):
        df_tmp = data[(data['timestamp'] > time_list[t]) & (data['timestamp'] <= time_list[t]+f_term)]
        mean = df_tmp[target_axis].mean()
        var  = df_tmp[target_axis].var()
        min  = df_tmp[target_axis].min()
        max  = df_tmp[target_axis].max()
        med  = df_tmp[target_axis].median()
        skew = df_tmp[target_axis].skew()
        kurt = df_tmp[target_axis].kurt()
        rms  = math.sqrt((df_tmp[target_axis]**2).mean())
        d_f  = LinearRe(df_tmp).coef_[0][0]
        entl = entropy(df_tmp)
        p_b  = power_band(df_tmp)
        f_list.append([time_list[t], mean, var, min, max, med, skew, kurt, rms, d_f, entl, p_b])
    mk_feature_val = pd.DataFrame(f_list, columns=['time', 'mean', 'variance', 'min', 'max', 'median', 'skewness', 'kurtosis', 'rms', 'dynamic_feature', 'entropy', 'power_band'])
    return mk_feature_val

def pycaret_train(file, model):
    ##########初期設定##########
    #@title グラフテーマ： ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none]
    template = 'plotly' #@param ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'] {allow-input: true}

    ##########データ読込み##########
    #@title csvファイル（UTF-8）を指定してください
    target = file
    
    
    data = pd.read_csv(target, header=4)
    data.columns = ['timestamp','STREAM','DAMAXIS','U-D']
    
    ##########Pycaret実装##########
    feature_val = mk_feature_val(data, f_nom_train)
    feature_val.set_index('time', drop=True, inplace=True)
    data = data[["timestamp",target_axis]]
    data.set_index('timestamp', drop=True, inplace=True)
    
    #@title Pycaretセットアップ - セットアップが完了したらEnterキーを押してください
    # init setup
    print(feature_val)
    s = setup(feature_val, normalize = True, normalize_method = 'zscore' , session_id = 123)
    time.sleep(3)
    pyautogui.press('enter')
    
    ##########異常検知のモデル選択と実行##########
    #++++++++++++++学習++++++++++++++
    # train model
    AD_Pycaret = create_model(model, fraction = 0.1, verbose = True, experiment_custom_tags = None)
    if model == 'iforest':
        AD_Pycaret.n_estimators=100 #default 100
        AD_Pycaret.max_samples='auto'  #default 'auto'
        AD_Pycaret.contamination=0.1   #default 0.1
        AD_Pycaret.max_features=1.0    #default 1.0
        AD_Pycaret.bootstrap=False     #default False
        AD_Pycaret.n_jobs=1            #default 1
        AD_Pycaret.behaviour='old'     #default 'old'
        AD_Pycaret.random_state=None   #default None
        AD_Pycaret.verbose=0           #default 0
    #model保存
    save_model(AD_Pycaret, 'model_{}'.format(model))
    
    AD_Pycaret_results = assign_model(AD_Pycaret)
    AD_Pycaret_results.to_csv('./#output/res_{}_train.csv'.format(model))
    
    #@title 選択モデルによる異常検知のグラフ表示
    # plot value on y-axis and date on x-axis
    fig, axes = plt.subplots(nrows=14, figsize=(80,40))
    axes[0].plot(data.index, data[target_axis], color="black") # 変換前
    axes[1].plot(AD_Pycaret_results.index, AD_Pycaret_results['Anomaly'], marker='o', color="r", linestyle='None')
    axes[2].plot(AD_Pycaret_results.index, AD_Pycaret_results['Anomaly_Score'], marker='o', color="r", linestyle='None')
    axes[3].plot(AD_Pycaret_results.index, AD_Pycaret_results['mean'], color="b")
    axes[4].plot(AD_Pycaret_results.index, AD_Pycaret_results['variance'], color="b")
    axes[5].plot(AD_Pycaret_results.index, AD_Pycaret_results['min'], color="b")
    axes[6].plot(AD_Pycaret_results.index, AD_Pycaret_results['max'], color="b")
    axes[7].plot(AD_Pycaret_results.index, AD_Pycaret_results['median'], color="b")
    axes[8].plot(AD_Pycaret_results.index, AD_Pycaret_results['skewness'], color="b")
    axes[9].plot(AD_Pycaret_results.index, AD_Pycaret_results['kurtosis'], color="b")
    axes[10].plot(AD_Pycaret_results.index, AD_Pycaret_results['rms'], color="b")
    axes[11].plot(AD_Pycaret_results.index, AD_Pycaret_results['dynamic_feature'], color="b")
    axes[12].plot(AD_Pycaret_results.index, AD_Pycaret_results['entropy'], color="b")
    axes[13].plot(AD_Pycaret_results.index, AD_Pycaret_results['power_band'], color="b")
    axes[0].set_ylabel(target_axis+'(gal)', fontsize=24)
    axes[1].set_ylabel('Anomaly', fontsize=24)
    axes[2].set_ylabel('Anomaly_Score', fontsize=24)
    axes[3].set_ylabel('mean(gal)', fontsize=24)
    axes[4].set_ylabel('variance', fontsize=24)
    axes[5].set_ylabel('min(gal)', fontsize=24)
    axes[6].set_ylabel('max(gal)', fontsize=24)
    axes[7].set_ylabel('median(gal)', fontsize=24)
    axes[8].set_ylabel('skewness', fontsize=24)
    axes[9].set_ylabel('kurtosis', fontsize=24)
    axes[10].set_ylabel('rms', fontsize=24)
    axes[11].set_ylabel('dynamic_feature', fontsize=24)
    axes[12].set_ylabel('entropy', fontsize=24)
    axes[13].set_ylabel('power_band', fontsize=24)
    axes[13].set_xlabel('time(s)', fontsize=24)
    for i in range(13):
        axes[i].tick_params(labelbottom=False)
    plt.savefig('./#output/res_{}_train.png'.format(model))
    plt.close
    
def pycaret_test(file, model):
    ##########初期設定##########
    #@title グラフテーマ： ggplot2, seaborn, simple_white, plotly, plotly_white, plotly_dark, presentation, xgridoff, ygridoff, gridon, none]
    template = 'plotly' #@param ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'] {allow-input: true}

    ##########データ読込み##########
    #@title csvファイル（UTF-8）を指定してください
    target = file
    
    data = pd.read_csv(target, header=4)
    data.columns = ['timestamp','STREAM','DAMAXIS','U-D']
    
    ##########Pycaret実装##########
    feature_val = mk_feature_val(data, f_nom_test)
    feature_val.set_index('time', drop=True, inplace=True)
    data = data[["timestamp",target_axis]]
    data.set_index('timestamp', drop=True, inplace=True)
    
    #@title Pycaretセットアップ - セットアップが完了したらEnterキーを押してください
    # init setup
    #s = setup(feature_val, normalize = True, normalize_method = 'zscore' , session_id = 123)
    #time.sleep(3)
    #pyautogui.press('enter')
    
    ##########異常検知のモデル選択と実行##########
    #++++++++++++++検証++++++++++++++
    AD_Pycaret = load_model('model_{}'.format(model))
    
    AD_Pycaret_results = predict_model(AD_Pycaret, data=feature_val)
    AD_Pycaret_results.to_csv('./#output/res_{}_test.csv'.format(model))

    #@title 選択モデルによる異常検知のグラフ表示
    # plot value on y-axis and date on x-axis
    fig, axes = plt.subplots(nrows=14, figsize=(80,40))
    axes[0].plot(data.index, data[target_axis], color="black") # 変換前
    axes[1].plot(AD_Pycaret_results.index, AD_Pycaret_results['Anomaly'], marker='o', color="r", linestyle='None')
    axes[2].plot(AD_Pycaret_results.index, AD_Pycaret_results['Anomaly_Score'], marker='o', color="r", linestyle='None')
    axes[3].plot(AD_Pycaret_results.index, AD_Pycaret_results['mean'], color="b")
    axes[4].plot(AD_Pycaret_results.index, AD_Pycaret_results['variance'], color="b")
    axes[5].plot(AD_Pycaret_results.index, AD_Pycaret_results['min'], color="b")
    axes[6].plot(AD_Pycaret_results.index, AD_Pycaret_results['max'], color="b")
    axes[7].plot(AD_Pycaret_results.index, AD_Pycaret_results['median'], color="b")
    axes[8].plot(AD_Pycaret_results.index, AD_Pycaret_results['skewness'], color="b")
    axes[9].plot(AD_Pycaret_results.index, AD_Pycaret_results['kurtosis'], color="b")
    axes[10].plot(AD_Pycaret_results.index, AD_Pycaret_results['rms'], color="b")
    axes[11].plot(AD_Pycaret_results.index, AD_Pycaret_results['dynamic_feature'], color="b")
    axes[12].plot(AD_Pycaret_results.index, AD_Pycaret_results['entropy'], color="b")
    axes[13].plot(AD_Pycaret_results.index, AD_Pycaret_results['power_band'], color="b")
    axes[0].set_ylabel(target_axis+'(gal)', fontsize=24)
    axes[1].set_ylabel('Anomaly', fontsize=24)
    axes[2].set_ylabel('Anomaly_Score', fontsize=24)
    axes[3].set_ylabel('mean(gal)', fontsize=24)
    axes[4].set_ylabel('variance', fontsize=24)
    axes[5].set_ylabel('min(gal)', fontsize=24)
    axes[6].set_ylabel('max(gal)', fontsize=24)
    axes[7].set_ylabel('median(gal)', fontsize=24)
    axes[8].set_ylabel('skewness', fontsize=24)
    axes[9].set_ylabel('kurtosis', fontsize=24)
    axes[10].set_ylabel('rms', fontsize=24)
    axes[11].set_ylabel('dynamic_feature', fontsize=24)
    axes[12].set_ylabel('entropy', fontsize=24)
    axes[13].set_ylabel('power_band', fontsize=24)
    axes[13].set_xlabel('time(s)', fontsize=24)
    for i in range(13):
        axes[i].tick_params(labelbottom=False)
    plt.savefig('./#output/res_{}_test.png'.format(model))
    plt.close

if __name__ == '__main__':
    main()

            
