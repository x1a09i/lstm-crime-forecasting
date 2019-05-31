from keras import optimizers
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.layers import LSTM
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties
import numpy as np


def run_lp(crime_type, year):
    # ファイル読み込み
    df_supervised = read_csv('data/lstm/' + str(year) + '/thief-' + crime_type + '.csv', header=0, index_col=0)
    supervised_values = df_supervised.values
    # 教師データ・テストデータで切り分ける
    n_train = 20000
    train = supervised_values[:n_train, :]
    test = supervised_values[n_train:, :]
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]
    # LSTMに入力する形状に変換
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    # モデル構築
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mae', optimizer=adam)

    history = model.fit(train_x, train_y, epochs=50, batch_size=1024, validation_data=(test_x, test_y), verbose=2,
                        shuffle=False)
    # 学習過程を描画する
    fig = pyplot.figure(figsize=(8, 6))
    fp = FontProperties(fname=r'C:\WINDOWS\Fonts\msgothic.ttc', size=14)
    pyplot.plot(history.history['loss'], label='教師データ')
    pyplot.plot(history.history['val_loss'], '--', label='テストデータ')
    pyplot.xlabel('エポック数', fontproperties=fp)
    pyplot.ylabel('損失関数の出力', fontproperties=fp)
    # pyplot.title('損失関数の推移', fontproperties=fp)
    pyplot.legend(prop=fp)
    pyplot.show()

    fig.savefig('result/learn.svg', dpi=300, facecolor='None', edgecolor='None', transparent=True, format='svg')

    # スケール戻す準備
    df_year = read_csv('data/raw/' + str(year) + '/thief-' + crime_type + '.csv', header=0, index_col=[0, 1])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit_transform(df_year.values.astype('float32'))
    # 予測プロセス
    yhat = model.predict(test_x)
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    # 予測値
    inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # 実際の値
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # 予測結果をプロット
    fig = pyplot.figure(figsize=(8, 10))
    pyplot.subplot(2, 1, 1)
    # pyplot.ylim([np.min(inv_y[-1000:] - 0.1), np.max(inv_y[-1000:]) + 0.1])
    pyplot.ylim([np.min(inv_yhat[-1000:] - 0.1), np.max(inv_y[-1000:]) + 0.1])
    pyplot.plot(inv_yhat[-1000:], color='red', label='予測の値')
    pyplot.ylabel('認知件数', fontproperties=fp)
    pyplot.legend(prop=fp)
    pyplot.subplot(2, 1, 2)
    pyplot.plot(inv_y[-1000:], color='blue', label='実際の値')
    pyplot.ylabel('認知件数', fontproperties=fp)
    pyplot.legend(prop=fp)
    pyplot.show()
    fig.savefig('result/predict.svg', dpi=300, facecolor='None', edgecolor='None', transparent=True, format='svg')
    # RMSEの値を計算
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    # モデルの保存
    open('result/' + crime_type + str(year) + '.json', "w").write(model.to_json())
    # 学習済みの重みを保存
    model.save_weights('result/' + crime_type + str(year) + '.hdf5')

    # 入力層の重みをプロットする
    w1 = model.layers[0].get_weights()[0]
    pyplot.figure()
    pyplot.plot((w1 ** 2).mean(axis=1), 'o-')
    pyplot.show()


run_lp('akisu', 2016)
