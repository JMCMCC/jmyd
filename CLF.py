import pandas as pd
import numpy as np
import math
from math import radians, cos, sin, asin, sqrt
dataset = pd.read_csv('F://mrtext//6801251.csv')#保存csv时候编码类型注意保存为utf-8，若已保存，用记事本打开，另存为修改编码类型
#dataset.ix[:20]

#mr采样数据中经纬度范围最大最小值额外外扩500m
df1=dataset['经度']
#print(df1)
lon_max=df1.max()
lon_min=df1.min()
df2=dataset['纬度']
lat_max=df2.max()
lat_min=df2.min()
print(lon_max)
print(lon_min)
print(lat_max)
print(lat_min)
#根据（lat2-lat1）*40000/360=X(km)
#当X=500m时，Dlat=0.0045度
#不同经度的间距因其所处的纬度不同而不同
#（lon2-lon1）*cos（lat） *40000/360=Y（km）
#当Y=500m时，Dlon=0.0045/cos(lat)



latMax=lat_max+0.0045#latMax,latMin,lonMax,lonMin为输入四个值
latMin=lat_min-0.0045
latMaxR=math.radians(latMax)
latMinR=math.radians(latMin)
lonMax=lon_max+0.0045/cos(latMaxR)#采用纬度中间值做运算好还是采用各自值做运算好？
lonMin=lon_min-0.0045/cos(latMinR)
print(latMin)
print(latMax)
print(lonMin)
print(lonMax)

#50*50m per格，计算输入经纬度范围内，南北、东西距离，换算成50米为多少格
#南北走向格数
rowF=(latMax-latMin)/0.00045
#rowF为浮点数
row=math.ceil(rowF)#向上取整
print(row)

#东西走向格数

lat=(latMaxR+latMinR)/2#取平均纬度计算更为准确
indexF=(lonMax-lonMin)*cos(lat)/0.00045
index=math.ceil(indexF)
print(index)



#dataset中经纬度换算成栅格序号
r=0.00045/cos(lat)#经度一小格的经度差，lat为平均纬度
				  #注意！！！此处lat仍是使用输入数据算出来的，所以需要统一lat的赋值！！！
i=0.00045#纬度一小格的纬度差
dataset['A']=None
L1=[]
L2=[]
#遍历dataframe中row纬度在第几格
for long in df1:
    #print(long)
    a=(long-lonMin)/r
    #print(a)
    A=math.ceil(a)
    #print(A)
    L1.append(A)
#print(L1)
dataset['A']=L1
#print(dataset)
    
#遍历dataframe中row经度在第几格
for lati in df2:
    b=(lati-latMin)/i
    B=index+1-math.ceil(b)#注意！！！此处使用的index仍是输入数据算出来的，所以存在负数！！！
    L2.append(B)
#print(L2)
dataset['B']=L2
print(dataset)


#定义函数，通过经纬度在栅格位置A，B，算栅格序号Y
def grid (A,B):
    return index*(B-1)+A

dataset['Y']=dataset.apply(lambda dataset: grid(dataset['A'],dataset['B']),axis=1)
print(dataset)


#整合输入集X，输出集Y
X=dataset.loc[:,('rsrp','rsrq','频点','ta','phr','ulsinr')]
#ci数据格式有误，无法识别
#选择多列方法！！！！！使用loc,格式loc[开头行：结尾行（：为全选），（‘列名’）]
X=X.values
Y=dataset["Y"].values

#决策树模型
#from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
#clf = DecisionTreeClassifier(random_state=14)
#scores= cross_val_score(clf,X,Y,scoring="accuracy")
#print('Accuracy:{0:.1f}%'.format(np.mean(scores)*100))#np.mean()是计算矩阵均值


#随机森林模型
from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(random_state=14)
#scores = cross_val_score(clf,X,Y,scoring = 'accuracy')
#print('Accuracy: {0:.1f}%'.format(np.mean(scores)*100))


#随机森林（使用GridSearchCV搜索最佳参数，数据量不宜太大，耗时间太久）
from sklearn.model_selection import GridSearchCV
parameter_space={
	'max_features': [0.2,'auto'],#寻求原因ValueError: max_features must be in (0, n_features]，
	                             #此处使用0.2，允许每个随机森林的子树可以利用变量（特征）数的20％。
	                             #如果想考察的特征x％的作用， 我们可以使用“0.X”的格式
	'n_estimators':[100,],
	'criterion':['gini','entropy'],
	'min_samples_leaf':[2,4,6],
}
clf = RandomForestClassifier(random_state=14)
grid2 = GridSearchCV(clf,parameter_space)
grid2.fit(X,Y)


print('Accuracy:{0:.1f}%'.format(grid2.best_score_*100))