import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF
import statsmodels.api as sm
import datetime
from itertools import product
from datetime import  timedelta
import calendar
import warnings                                  # do not disturbe mode
warnings.filterwarnings('ignore')
import PyQt5
matplotlib.use('Qt5Agg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'SimHei'#支持显示中文
plt.rcParams['axes.unicode_minus'] = False #支持显示中文

path = '代码数据'
bitcoin = pd.read_csv(path+'/BCHAIN-MKPRU.csv', index_col=['Date'], parse_dates=True)#读取比特币数据
gold = pd.read_csv(path+'/LBMA-GOLD.csv', index_col=['Date'], parse_dates=True)#读取黄金数据
# data=gold.dropna()#删除缺失值，默认行
# df = gold["USD (PM)"].fillna( method='ffill',inplace = True) #使用上一个值来进行替代，inplace意为直接在原始数据中进行修改
df3=bitcoin
# df1[df1["USD (PM)"].isnull()]#判断缺失数据


# gold[gold["USD (PM)"].isnull()] #判断缺失值
gold["USD (PM)"].fillna( method='ffill',inplace = True) #使用上一个值来进行替代，inplace意为直接在原始数据中进行修改
#创建DataFrame数据，包括index列和value列，其中index列为日期，但是格式为string格式
data = gold#pd.DataFrame(data={'index':['2021-1-01','2020-10-10','2020-10-17','2020-10-15'],'value':range(4)})


date_list1=[]
#计算最小日期和最大日期
date_start =data.index.min().strftime('%Y-%m-%d')
date_end =data.index.max().strftime('%Y-%m-%d')
#date_end - date_start#
#根据最小日期和最大日期，计算日期间隔，由于date_start和date_end为string类型，因此需要先更改为日期类型
delta =datetime.datetime.strptime(date_end, "%Y-%m-%d")-datetime.datetime.strptime(date_start, "%Y-%m-%d")

for i in range(1,delta.days+1):
    date =(datetime.datetime.strptime(date_end, "%Y-%m-%d")-datetime.timedelta(days=i)).strftime('%Y-%m-%d')
    #如果数据缺失，则补数
    if(date not in data.index.strftime('%Y-%m-%d')):
        date_list1.append(datetime.datetime.strptime(date,('%Y-%m-%d')))
    future = pd.DataFrame(index=date_list1, columns= data.columns)
    df_month2 = pd.concat([data, future])
#按照日期列排序
dd=df_month2.sort_index(ascending=True)#根据日期排序
#工作日的价格为上一天即这个周五的价格
dd["USD (PM)"].fillna( method='ffill',inplace = True) #使用上一个值来进行替代，inplace意为直接在原始数据中进行修改
gold=dd



# 按照月，季度，年来统计
df=dd#黄金数据统计
dd2=bitcoin
df1=dd2#比特币数据统计
df_month = df.resample('M').mean()
df_month#跟df相比  数据少了很多 因为是按照一个月统计一次  按照每个月的平均值来统计

df_Q = df.resample('Q-DEC').mean()
df_Q#这个是按照季度

df_year = df.resample('A-DEC').mean()
df_year# 这个是按照年份 一年统计一次
#比特币
df_month1 = df1.resample('M').mean()
df_month1#跟df相比  数据少了很多 因为是按照一个月统计一次  按照每个月的平均值来统计

df_Q1 = df1.resample('Q-DEC').mean()
df_Q1#这个是按照季度

df_year1 = df1.resample('A-DEC').mean()
df_year1# 这个是按照年份 一年统计一次



#黄金价格可视化
# 按照天，月，季度，年来显示沪市指数的走势
fig = plt.figure(figsize=[15, 7])
plt.suptitle('Gold Daily Price', fontsize=20) 
plt.subplot(221)
plt.plot(df["USD (PM)"], '-', label='byDay')
plt.legend()
plt.subplot(222)
plt.plot(df_month["USD (PM)"], '-', label='byMonth')
plt.legend()
plt.subplot(223)
plt.plot(df_Q["USD (PM)"], '-', label='bySeason')
plt.legend()
plt.subplot(224)
plt.plot(df_year["USD (PM)"], '-', label='byYear')
plt.legend()
plt.show()



#比特币价格可视化
# 按照天，月，季度，年来显示沪市指数的走势
fig = plt.figure(figsize=[15, 7])
plt.suptitle('Bitcoin Daily Price', fontsize=20)
plt.subplot(221)
plt.plot(df1["Value"], '-', label='byDay')
plt.legend()
plt.subplot(222)
plt.plot(df_month1["Value"], '-', label='byMonth')
plt.legend()
plt.subplot(223)
plt.plot(df_Q1["Value"], '-', label='bySeason')
plt.legend()
plt.subplot(224)
plt.plot(df_year1["Value"], '-', label='byYear')
plt.legend()
plt.show()



bitcoin["diff1"] = bitcoin["Value"].diff(1).dropna()#1阶差分
bitcoin["diff2"] = bitcoin["Value"].diff(2).dropna()#2阶差分
# data["diff3"] = data["diff2"].diff(1).dropna()#2阶差分
# data["diff4"] = data["diff3"].diff(1).dropna()#2阶差分
# fig=plt.figure(figsize=(12,8))
bitcoin1 = bitcoin.loc[:,["Value","diff1","diff2"]]#,"diff3","diff4"
bitcoin1.plot(subplots=True, figsize=(12, 8),title="Bitcoin data difference diagram",fontsize=15)
# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 12})
# plt.yticks(fontproperties = 'Times New Roman', size = 20)
#fontsize设置坐标轴字体大小  figsize设置图片大小，subplots=True根据dataframe的columns绘制子图
gold["diff1"] = gold["USD (PM)"].diff(1).dropna()#1阶差分
gold["diff2"] = gold["USD (PM)"].diff(2).dropna()#2阶差分
# data["diff3"] = data["diff2"].diff(1).dropna()#2阶差分
# data["diff4"] = data["diff3"].diff(1).dropna()#2阶差分
gold1 = gold.loc[:,["USD (PM)","diff1","diff2"]]#,"diff3","diff4"
gold1.plot(subplots=True, figsize=(12, 8),title="Gold data difference diagram",fontsize=15)

# plt.legend(prop={'family' : 'Times New Roman', 'size'   : 12})

#从图中可以看出一阶差分比二阶和原数据更平稳  故参数d=1



#ARIMA模型分析：
#先把ACF图和PACF图画出来看看：
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(bitcoin["Value"].diff(2).iloc[1:len(list(bitcoin["Value"]))-1].dropna(),lags=24,ax=ax1) # 注意：要去掉第1个空值                             
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(bitcoin["Value"].diff(2).iloc[1:len(list(bitcoin["Value"]))-1].dropna(),  lags=24,ax=ax2)# 注意：要去掉第1个空值
#判断：ACF图在2之后截尾，而PACF拖尾。bitcoin模型可以由MA(2)→ARIMA(0,1,2).
#自相关（ACF）图和偏自相关（PACF）图



print("单位根检验:\n")
print("比特币:",'\n',ADF(bitcoin.diff1.dropna()),'\n')  
print("黄金:",'\n',ADF(gold.diff1.dropna())) 


#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print("黄金白噪声检验:\n")
a=acorr_ljungbox(gold.diff1.dropna(), lags = [i for i in range(1,12)],boxpierce=True)
print(a,'\n')
print("比特币噪声检验:\n")
a=acorr_ljungbox(bitcoin.diff1.dropna(), lags = [i for i in range(1,12)],boxpierce=True)
print(a,'\n')