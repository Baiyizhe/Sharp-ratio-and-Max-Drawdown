import numpy as np
import pandas as pd


class Indicator(object):

    def __init__(self, data):
        self.data = np.array(data[0])
        self.df = data[0]

    def final_value(self):
        return self.data[-1]

    def MaxDrawdown(self):
        index_j = np.argmax(np.maximum.accumulate(self.data) - self.data)  # 结束位置
        if index_j == 0:
            return 0
        index_i = np.argmax(self.data[:index_j])  # 开始位置
        d = (self.data[index_i] - self.data[index_j]) / self.data[index_i]  # 最大回撤
        return d

    def sharpe_ratio(self):
        '''夏普比率'''
        returns = self.df - self.df.shift(1)  # 每日收益
        average_return = np.mean(returns)
        return_stdev = np.std(returns)

        AnnualRet = average_return * 252  # 默认252个工作日
        AnnualVol = return_stdev * np.sqrt(252)
        sharpe_ratio = (AnnualRet - 0.02) / AnnualVol  # 默认无风险利率为0.02
        return (sharpe_ratio)


if __name__ == '__main__':
    data = pd.read_csv('./data/test.csv', header=None)
    indicator = Indicator(data)
    MDD = indicator.MaxDrawdown()
    sharp_ratio = indicator.sharpe_ratio()
    print('sharp_ratio  :  ', sharp_ratio)
    print('MDD :  ', MDD)
    print('final value  : ', indicator.final_value())