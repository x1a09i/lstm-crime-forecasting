from keras.engine.saving import model_from_json
from matplotlib import pyplot
from matplotlib.font_manager import FontProperties


def plot_weight(crime_type):
    # 学習済みのモデルを読み込み
    model_2014 = model_from_json(open('result/' + crime_type + '2014.json', 'r').read())
    model_2015 = model_from_json(open('result/' + crime_type + '2015.json', 'r').read())
    model_2016 = model_from_json(open('result/' + crime_type + '2016.json', 'r').read())
    model_2017 = model_from_json(open('result/' + crime_type + '2017.json', 'r').read())
    model_2018 = model_from_json(open('result/' + crime_type + '2018.json', 'r').read())
    # 重みを読み込む
    model_2014.load_weights('result/' + crime_type + '2014.hdf5')
    model_2015.load_weights('result/' + crime_type + '2015.hdf5')
    model_2016.load_weights('result/' + crime_type + '2016.hdf5')
    model_2017.load_weights('result/' + crime_type + '2017.hdf5')
    model_2018.load_weights('result/' + crime_type + '2018.hdf5')
    # 入力層の重みを抽出
    w_2014 = model_2014.layers[0].get_weights()[0]
    w_2015 = model_2015.layers[0].get_weights()[0]
    w_2016 = model_2016.layers[0].get_weights()[0]
    w_2017 = model_2017.layers[0].get_weights()[0]
    w_2018 = model_2018.layers[0].get_weights()[0]
    # 入力層の重みをプロットする
    fp = FontProperties(fname=r'C:\WINDOWS\Fonts\msgothic.ttc', size=14)
    fig = pyplot.figure(figsize=(12, 6))
    pyplot.plot((w_2014 ** 2).mean(axis=1), 'o-', label='2014')
    pyplot.plot((w_2015 ** 2).mean(axis=1), '^-', label='2015')
    pyplot.plot((w_2016 ** 2).mean(axis=1), 's-', label='2016')
    pyplot.plot((w_2017 ** 2).mean(axis=1), 'x-', label='2017')
    pyplot.plot((w_2018 ** 2).mean(axis=1), 'D-', label='2018')
    pyplot.xlabel('入力ユニット', fontproperties=fp)
    pyplot.ylabel('重みの平均値', fontproperties=fp)
    pyplot.legend(prop=fp)
    pyplot.show()
    fig.savefig('result/weight.svg', dpi=300, facecolor='None', edgecolor='None', transparent=True, format='svg')


plot_weight('bicycle')
