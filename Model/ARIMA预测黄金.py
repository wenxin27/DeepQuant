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
temp=pd.read_csv(path+"/C题处理后的中间文件2.csv")
print("data:", data)


start_timestamp=to_timestamp(data.iloc[0,0])
for i in range(data.shape[0]):
    data.iloc[i,0]=(to_timestamp(data.iloc[i,0])-start_timestamp)/86400
print("data:", data)

data = data["黄金价值"]
print("data:", data)
data = data.dropna().to_frame()
print("data:", data)



#data.plot()
#plt.show() # 1.时序图显示该序列具有明显的单调递增趋势，可以判断为是非平稳序列

print("data.shape", data.shape)
#plot_acf(data).show() # 2.自相关系数图显示自相关系数长期大于0，说明序列间具有很强的长期相关性

result =ADF(data[u'黄金价值'])
print(f'p-value: {result[1]}') # 3.p-value和0.05比较，大于的话就是非平稳序列



D_data = data.diff().dropna()
D_data.columns = [u'黄金价值差分']
#D_data.plot() # 时序图
#plt.show()
#plot_acf(D_data).show() # 自相关图
#plt.show()  #14
#plot_pacf(D_data).show() # 偏自相关图
#plt.show()  #13

result =ADF(D_data[u'黄金价值差分'])
print(f'p-value: {result[1]}') # 黄金价值差分1.01e-13<0.05

print("D_data:",D_data)
print("type:",type(D_data))
D_data.reset_index(inplace=True)
D_data.drop('index', axis=1, inplace=True)
print("D_data:",D_data)


print(u'差分序列的白噪声检验结果为: ', acorr_ljungbox(D_data, lags=1)) # lb_pvalue = 0.424308, p-value>0.05为白噪声



data.reset_index(inplace=True)
print("data:",data)
data.drop('index', axis=1, inplace=True)
print("data:", data)








model = ARIMA(data[u'黄金价值'], order=(2,1,2)).fit()  # 建立ARIMA(2,1,2)模型 p,d,q 注意这里要输入的是原始数据，不是差分数据
print('模型报告: \n', model.summary())
print('未来5天: \n', model.forecast(5)) 

pred = model.predict(start=1, end=data.shape[0]-1,typ='levels') 
print("pred:", pred)
pred = pred.to_frame()
print("pred:", pred)
print("type:", type(pred))



book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet("ARIMA预测黄金", cell_overwrite_ok=True)
col = ("日期","标签","真实值","预测值","误差")
for i in range(0, 5):
    sheet.write(0, i, col[i])
sheet.write(1,0,0)
sheet.write(1,1,1)  #不可交易黄金


sheet.write(2,0,1)  #可交易黄金
sheet.write(2,1,0)
sheet.write(2,2,1324.6)
sheet.write(2,3,1324.6)
sheet.write(2,4,0)


times = 0
for i in range(0, 1824):

    if(temp.values[i+2][2]==1):
        sheet.write(i + 3, 0, i+3)
        sheet.write(i + 3, 1, temp.values[i+2][2])
    else:

        sheet.write(i + 3, 0, i+3)
        sheet.write(i + 3, 1, temp.values[i+2][2])
        sheet.write(i + 3, 2, temp.values[i+2][3])
        sheet.write(i + 3, 3, pred.values[times][0])
        sheet.write(i + 3, 4, abs(temp.values[i+2][3]-pred.values[times][0]))
        times += 1


book.save("ARIMA预测黄金.xls")


