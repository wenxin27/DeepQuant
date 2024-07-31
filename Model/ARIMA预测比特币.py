import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
import xlwt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def to_timestamp(date):
    return int(time.mktime(time.strptime(date,"%m/%d/%y")))


path="代码数据"
data=pd.read_csv(path+"/C题处理后的中间文件2.csv")
print("data", data)


start_timestamp=to_timestamp(data.iloc[0,0])
for i in range(data.shape[0]):
    data.iloc[i,0]=(to_timestamp(data.iloc[i,0])-start_timestamp)/86400
print("data", data)

data = data[["日期(月/日/年)","比特币价值"]]
print("data", data)
data.set_index('日期(月/日/年)', inplace=True)

print("data[u'比特币价值']", data[u'比特币价值'])


#data.plot()
#plt.show() # 1.时序图显示该序列具有明显的单调递增趋势，可以判断为是非平稳序列

print("data.shape", data.shape)
#plot_acf(data).show() # 2.自相关系数图显示自相关系数长期大于0，说明序列间具有很强的长期相关性

result =ADF(data[u'比特币价值'])
print(f'p-value: {result[1]}') # 3.p-value和0.05比较，大于的话就是非平稳序列



D_data = data.diff().dropna()
D_data.columns = [u'比特币价值差分']
#D_data.plot() # 时序图
#plt.show()
#plot_acf(D_data).show() # 自相关图
#plt.show()  #14
#plot_pacf(D_data).show() # 偏自相关图
#plt.show()  #13

result =ADF(D_data[u'比特币价值差分'])
print(f'p-value: {result[1]}') # 比特币价值差分1.01e-13<0.05


print(u'差分序列的白噪声检验结果为: ', acorr_ljungbox(D_data, lags=1)) # lb_pvalue = 0.001023 <0.05,说明差分序列为平稳非白噪声序列




model = ARIMA(data[u'比特币价值'], order=(4,1,4)).fit()  # 建立ARIMA(4,1,4)模型 p,d,q 注意这里要输入的是原始数据，不是差分数据
print('模型报告: \n', model.summary())
print('未来5天: \n', model.forecast(5)) 

pred = model.predict(start=1, end=data.shape[0]-1,typ='levels') 
print("pred:", pred)
pred = pred.to_frame()
print("pred:", pred)
print("type:", type(pred))



book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet("ARIMA预测比特币", cell_overwrite_ok=True)
col = ("日期","预测值","真实值","误差")  


for i in range(0, 4):
    sheet.write(0, i, col[i])

sheet.write(1,0,0)
sheet.write(1,1,621.25)
sheet.write(1,2,621.25)
sheet.write(1,3,0)

for i in range(0, 1825):
    sheet.write(i+2,0,i+1 )
    sheet.write(i+2,1,pred.values[i][0])
    sheet.write(i+2,2,data[u'比特币价值'][i+1])
    sheet.write(i+2,3,abs(data[u'比特币价值'][i+1]-pred.values[i][0]))
book.save("ARIMA预测比特币.xls")