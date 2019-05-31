from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from os import path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 正解ラベルの時系列データを作る
def series_to_supervised(data, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 入力シーケンス (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 出力（予測）シーケンス (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # データ結合
    agg = concat(cols, axis=1)
    agg.columns = names
    # 欠損値のある行を削除
    agg.dropna(inplace=True)
    return agg


# 教師あり学習のデータ形式に変換
def convert_to_lstm(crime_type, year, bacth_size):
    if crime_type == 0:
        path_str = 'data/raw/thief-akisu.csv'
    elif crime_type == 1:
        path_str = 'data/raw/thief-bicycle.csv'
    else:
        path_str = 'data/raw/thief-manbiki.csv'
    df_raw = read_csv(path_str, header=0)
    # 年単位で抽出する
    df_year = df_raw[df_raw.year_month.between(str(year) + '-01-01', str(year) + '-12-01')]
    df_year.set_index(['key_code', 'year_month'], drop=True, inplace=True)
    # スケール変換のために出力しておく
    df_year.to_csv('data/raw/' + str(year) + '/' + path.basename(path_str))
    # 標準化
    year_values = df_year.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    values_scaled = scaler.fit_transform(year_values)
    df_year_scaled = pd.DataFrame(values_scaled)
    # 地域単位で学習データを作り出す
    df_output = pd.DataFrame()
    for i_conv in range(0, len(df_year_scaled.index), bacth_size):
        crime_value = df_year_scaled[i_conv:i_conv+bacth_size].values
        # 教師あり学習の入力データに変換
        df_supervised = series_to_supervised(crime_value, 6, 1)
        df_output = df_output.append(df_supervised)
    # インデックス振り直す
    df_output.reset_index(drop=True, inplace=True)
    # 1から始めるようにする
    df_output.index = df_output.index + 1
    # 名をつける
    df_output.index.name = 'id'
    # 年間フォルダーに出力
    df_output.to_csv('data/lstm/' + str(year) + '/' + path.basename(path_str))


# 罪種ごとに変換
for i in range(3):
    for j in range(2014, 2019):
        convert_to_lstm(i, j, 59)
