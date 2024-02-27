# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:46:57 2023

@author: Shraddha Ghuge
"""

'''Problem Statement: -
Kitabi Duniya, a famous book store in India, 
which was established before Independence, 
the growth of the company was incremental year by year, 
but due to online selling of books and wide spread 
Internet access its annual growth started to collapse, 
seeing sharp downfalls, you as a Data Scientist help 
this heritage book store gain its popularity back and
 increase footfall of customers and provide ways the
 business can improve exponentially, apply Association RuleAlgorithm, 
 explain the rules, and visualize the graphs for clear
 understanding of solution.
 
 
********1.) Books.csv******

'''

'''
#Business Objective:
    
maximize: Identify and promote book combinations that are 
frequently purchased together to increase cross-selling opportunities.

Minimize: Increase sales and revenue by promoting popular book categories 

Constraints: The business needs to address online competition.
Strategies should include both online and offline components to capture a broader market.
'''

'''
DataDictonary:

Nominal Data:

'ChildBks': Children's books category.
'YouthBks': Youth books category.
'CookBks': Cookbooks category.
'RefBks': Reference books category.
'ArtBks': Art books category.
'GeogBks': Geography books category.
'ItalCook': Italian Cookbooks category.
'ItalAtlas': Italian Atlases category.
'ItalArt': Italian Art books category.
'Florence': Possibly a location or specific book related to Florence.

Ordinal Data:

'DoItYBks': Do-it-yourself books category.
'''

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('book.csv')
df

####################################
df.columns
'''
['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 'ArtBks',
       'GeogBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence']
'''
###################################
df.shape
# the dataset contain 2000 rows and 11 columns
###################################
df.dtypes
'''
ChildBks     int64
YouthBks     int64
CookBks      int64
DoItYBks     int64
RefBks       int64
ArtBks       int64
GeogBks      int64
ItalCook     int64
ItalAtlas    int64
ItalArt      int64
Florence     int64

The datatype is of numeric type there is no need of encoding
'''
######################################
a=pd.isnull(df)
a.sum()
'''
ChildBks     0
YouthBks     0
CookBks      0
DoItYBks     0
RefBks       0
ArtBks       0
GeogBks      0
ItalCook     0
ItalAtlas    0
ItalArt      0
Florence     0
dtype: int64

As there is no null value in the dataset
'''
#####################################
q=df.value_counts()
####################################
# Five Number Summary
v=df.describe()
# The mean value is near to zero and also the standard deviation is a;dp
# near to zero and the meadian value for the all datapoints is zero
df.info()
'''
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   ChildBks   2000 non-null   int64
 1   YouthBks   2000 non-null   int64
 2   CookBks    2000 non-null   int64
 3   DoItYBks   2000 non-null   int64
 4   RefBks     2000 non-null   int64
 5   ArtBks     2000 non-null   int64
 6   GeogBks    2000 non-null   int64
 7   ItalCook   2000 non-null   int64
 8   ItalAtlas  2000 non-null   int64
 9   ItalArt    2000 non-null   int64
 10  Florence   2000 non-null   int64
dtypes: int64(11)
'''
# This will give us the informationn about all the points

####################################
# Visualization of Data

# 1. Check for the outlier

sns.boxplot(df,x='ChildBks')
# No outlier 
sns.boxplot(df,x='YouthBks')
#There is one outlier 
sns.boxplot(df,x='CookBks')
# No Outlier
sns.boxplot(df,x='RefBks')
# There is one outlier
sns.boxplot(df)
# Observe that some columns contain  the outlier so we have to normalize it

#2. Pairplot
sns.pairplot(df)
# No Datapoints are corelated as the all the datapoints are in scatter form 

#3. Heatmap
corr=df.corr()
sns.heatmap(corr)
# The diagonal color of the heatmap is same as the datapoints folllow some pattern
# so we can use this data for the model building
############################################
#Normalization
#The data is numeric one so we have to perform normalization

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(v)
df_norm

b=df_norm.describe()

sns.boxplot(df_norm)
# No Outlier is remaining
# The all the quantile points are converted in the rande of 0-1
############################################
# Model Building
# Association Rules
from mlxtend.frequent_patterns import apriori,association_rules

data=pd.read_csv('book.csv')
data

# All the data is in properly separated form so no need to apply the encoding techique
# as it is already is in the form of numeric one

from collections import Counter
item_frequencies=Counter(data)

# Apriori algorithm
frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# This generate association rule for columns
# comprises of antescends,consequences

rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

# Visualize the rules
import networkx as nx
import matplotlib.pyplot as plt

# Create directed graph from the rules
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents')

# Draw the graph
fig, ax = plt.subplots(figsize=(14, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)
plt.title("Association Rules Network", fontsize=15)
plt.show()

###################################################

# the benefits/impact of the solution 
# By identifying books that are frequently purchased together,
# the bookstore can create curated bundles or recommendations, enhancing the overall 
# shopping experience for customers.
# By using this association rule we can stratergically placed the books together to encourage
# the customer to purchased more items which will help to increased the overall revenue


********************

***  2.  Groceries.csv ********
'''
Problem Statement: - 
The Departmental Store, has gathered the data of the products it sells on a Daily basis.
Using Association Rules concepts, provide the insights on the rules and the plots.
'''
from mlxtend.frequent_patterns import apriori,association_rules
# in this we are going use transaction data where size of each row is not same
# we canot use the pandas to load the data because it is in string form

#-----------------------read and split the all data/transaction

groceries=[]
#with open(r"D:\DATA SCIENCE\recommandation\groceries.csv") as f:groceries=f.read()
f= open(r"groceries.csv") 
groceries=f.read()
groceries
# it will split data in the comma separater form

groceries=groceries.split('\n')
groceries
#Earlier it was in the format of srin g it will convert it into the form of
# list
groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(','))
groceries_list
# it will separate items from each list so further we can separe it for support caculation
len(groceries_list)
#Out[19]: 9836

all_groceries_list=[i for item in groceries_list for i in item]
all_groceries_list
# we will get all the transactions/item 
# we will get 43368 items in various transaction
len(all_groceries_list)

#-----------------------Now let's count the frequency of each item 
#split frequency and item from dict

from collections import Counter
item_frequencies=Counter(all_groceries_list)
item_frequencies
#'frozen fruits': 12,
#'bags': 4,
#'cooking chocolate': 25,
#'sound storage medium': 1,
#'kitchen utensil': 4,
#'preservation products': 2,
#'': 1})

# item_frequencies will be contain key and dictionary 
# we want to sort it into count frequencies 
# means it will show he count of item purchased
# let us sort the frequencies in ascending order

item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
item_frequencies
#('soda', 1715),('rolls/buns', 1809),('other vegetables', 1903),('whole milk', 2513)]

#when we execute this,items frequencies will be in sorted form 
# item name with count

items=list(reversed([i[0] for i in item_frequencies]))
items
# This is the list comprehenssion it will give the items from dictionaries 

#l=[('soda', 1715),
#('rolls/buns', 1809),
#('other vegetables', 1903),
#('whole milk', 2513)]
#list(reversed([i[0] for i in l]))
#Out[50]: ['whole milk', 'other vegetables', 'rolls/buns', 'soda']

frequencies=list(reversed([i[1] for i in item_frequencies]))
frequencies
#l=[('soda', 1715),
#('rolls/buns', 1809),
#('other vegetables', 1903),
#('whole milk', 2513)]
#list(reversed([i[1] for i in l]))
#Out[54]: [2513, 1903, 1809, 1715]

#This will  give he frequencies of each items

#-----------------------------we will plot the frequencies

import matplotlib.pyplot as plt
plt.bar(height=frequencies[0:11],x=list(range(0,11)))
#here we just plot graph of 11 freq
#plt.xticks(list(range(0,11),items[0:11]))
plt.xlabel('items')
plt.ylabel('count')
plt.show()

#----------------Now we will convert it into dataframe
import pandas as pd
groceries_series=pd.DataFrame(pd.Series(groceries_list))
# Now we will get the the  dataframe of size 9836x1
# the last row of the dataframe is empty so we will remove it
groceries_series=groceries_series.iloc[:9836,:]
groceries_series
groceries_series.head(5)
# So it will remove the last row

# groceries_series having column name 0 so rename as Transaction
groceries_series.columns=['Transactions']
groceries_series

# So there is various elements which is separeted by , = we will seperate using
# * we will join it
x=groceries_series['Transactions'].str.join(sep='*')
x

# Now we will apply one-hot encoding to convert it into numeric form
x=x.str.get_dummies(sep='*')
x

# This is the data which we are going to apply for the Apriori algorithm
frequency_items=apriori(x,min_support=0.0075,max_len=4,use_colnames=True)
frequency_items
# You will get support value for 1,2,3,4 max items

# let us sort the support values
frequency_items.sort_values('support',ascending=False,inplace=True)
frequency_items

# This will sort the support the value in descending order 
# in EDA also there was same trend there it was a count
# and here it was support value

rules=association_rules(frequency_items,metric='lift',min_threshold=1)
# This generate association rule of size 1198x9 columns
# comprises of antescends,consequences
rules.head(20)
rules.columns
rules.sort_values('lift',ascending=False).head(10)



*******************

***  3.) my_movies.csv*******
'''
Problem Statement: - 
A film distribution company wants to target audience based on their likes and dislikes, you as a Chief Data Scientist Analyze the data and come up with different rules of movie list so that the business objective is achieved.

'''

'''
Business Objective:

Maximize: Audience engagement,profit

Minimize: Production time and production cost

ContraintsL: The business may face constraints related to market competition

'''

'''
DataFrame:
'Sixth Sense'
'Gladiator'
'LOTR1'
'Harry Potter1'
'Patriot'
'LOTR2'
'Harry Potter2'
'LOTR'
'Braveheart'
'Green Mile'

All the columns is of nominal there is no ordinal column in the dataset
'''
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('my_movies.csv')
df

####################################
df.columns
'''
['Sixth Sense', 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot',
       'LOTR2', 'Harry Potter2', 'LOTR', 'Braveheart', 'Green Mile']
'''
###################################
df.shape
# There is total 10 rows and 10 columns 
###################################
df.dtypes
'''
Sixth Sense      int64
Gladiator        int64
LOTR1            int64
Harry Potter1    int64
Patriot          int64
LOTR2            int64
Harry Potter2    int64
LOTR             int64
Braveheart       int64
Green Mile       int64
dtype: object
'''
# All the data in the dataset is of numeric type
####################################
df.info()
'''
#   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   Sixth Sense    10 non-null     int64
 1   Gladiator      10 non-null     int64
 2   LOTR1          10 non-null     int64
 3   Harry Potter1  10 non-null     int64
 4   Patriot        10 non-null     int64
 5   LOTR2          10 non-null     int64
 6   Harry Potter2  10 non-null     int64
 7   LOTR           10 non-null     int64
 8   Braveheart     10 non-null     int64
 9   Green Mile     10 non-null     int64
dtypes: int64(10)
'''
# It will show all the information about the columns
#####################################
a=df.describe()
a
# The mean and median are nearly same but shows some variation in the data
# the standard deviation is not equal to zero means the datapoints is in scatter form
#################################
# Check for the null values
v=df.isnull()
v.sum()
'''
Sixth Sense      0
Gladiator        0
LOTR1            0
Harry Potter1    0
Patriot          0
LOTR2            0
Harry Potter2    0
LOTR             0
Braveheart       0
Green Mile       0
dtype: int64
'''
# There is no null value in the dataset
##################################
# Visualization of data
# Plot the boxplot for outlier analysis
sns.boxplot(df,x='Sixth Sense')
# No Outlier
sns.boxplot(df,x='Gladiator')
# No Outlier
sns.boxplot(df,x='LOTR1')
# having one outlier
sns.boxplot(df)
# There is some outlier in the data

# Plot  the pairplot to understand behaviour
sns.pairplot(df)
# The datapoints are in scatter form

corr=df.corr()
sns.heatmap(corr)
# The diagonal colour of the heatmap is same so it follow some patern to understand
# the pattern we will perform some oerations
########################################
#Normalization
#The data is numeric one so we have to perform normalization

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(a)
df_norm

b=df_norm.describe()

sns.boxplot(df_norm)
# No Outlier is remaining
# The all the quantile points are converted in the rande of 0-1
###########################################
# Model Building
# Association Rules
from mlxtend.frequent_patterns import apriori,association_rules

data=pd.read_csv('my_movies.csv')
data

# All the data is in properly separated form so no need to apply the encoding techique
# as it is already is in the form of numeric one

from collections import Counter
item_frequencies=Counter(data)

# Apriori algorithm
frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# This generate association rule for columns
# comprises of antescends,consequences

rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

# Visualize the rules
import networkx as nx
import matplotlib.pyplot as plt

# Create directed graph from the rules
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents')

# Draw the graph
fig, ax = plt.subplots(figsize=(14, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)
plt.title("Association Rules Network", fontsize=15)
plt.show()

###################################################

# By using the association rule we will suggest the movies for the customer to 
# increase the viewers of the movies 
# Also by using this rules the revnue and the popularity of the movies
# will increases


*******************

********4.) myphonedata.csv*****
'''
Problem Statement: - 
A Mobile Phone manufacturing company wants to launch its three brand new phone into the market, but before going with its traditional marketing approach this time it want to analyze the data of its previous model sales in different regions and you have been hired as an Data Scientist to help them out, use the Association rules concept and provide your insights to the company’s marketing team to improve its sales.
4.) myphonedata.csv
'''

'''
Business Objective:

Maximize: The customer Satisfaction

Minimize: The product return

Cobnstrains: Resources
'''

'''
DataDictonary:

['red', 'white', 'green', 'yellow', 'orange', 'blue'] all the columns is of
nominal there is no ordinal data is present in the dataset
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('myphonedata.csv')
df

#################################
df.columns
'''
['red', 'white', 'green', 'yellow', 'orange', 'blue']
'''
##################################
df.head()
# It will show thw first five rows of the data
####################################
df.shape
# The dataset contain 11 rows and 6 columns
###################################
a=df.describe()
# The mean is near to zero and the mean and median diffrence is also not very
# big . The standard deviation is also near to zero
# so  we can say that the datapoints are scatter near the mradian
###################################
df.info()
'''
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   red     11 non-null     int64
 1   white   11 non-null     int64
 2   green   11 non-null     int64
 3   yellow  11 non-null     int64
 4   orange  11 non-null     int64
 5   blue    11 non-null     int64
dtypes: int64(6)
'''
###################################
df.dtypes
'''
red       int64
white     int64
green     int64
yellow    int64
orange    int64
blue      int64
dtype: object
'''
# The datatype of all columns is of numeric type so no need of encoding technique

######################################
# Find the missing values
df.isnull().sum()
'''
red       0
white     0
green     0
yellow    0
orange    0
blue      0
dtype: int64
'''
# There is no null  value in the daaset
######################################
# Visualize the dataset

#Plot the boxplot for the outlier analysis
sns.boxplot(df,x='red')
# No outlier 
sns.boxplot(df,x='white')
# No outlier
sns.boxplot(df,x='green')
# One outlier is present
sns.boxplot(df,x='yellow')
# One outlier present
sns.boxplot(df)
# There is three columns which contain the outliers 

# Plot a pairplot to understand the relationship between columns
sns.pairplot(df)
# The graphs does not show any relation as the datapoints are in scatter form

# To know more plot the heatmap
corr=df.corr()
sns.heatmap(corr)
# The heatmap showing some pattern of  the datapoins

############################################
#  As there is outliers present in the dataset so we normalize it using normalization technque


def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(a)
df_norm
# The mean and median diffrence is in the range of 0-1 and the standard deviation
# is also near to zero


b=df_norm.describe()

sns.boxplot(df_norm)
# No Outlier is remaining
# The all the quantile points are converted in the rande of 0-1
###########################################

# Model Building
# Association Rules
from mlxtend.frequent_patterns import apriori,association_rules

data=pd.read_csv('myphonedata.csv')
data

# All the data is in properly separated form so no need to apply the encoding techique
# as it is already is in the form of numeric one

from collections import Counter
item_frequencies=Counter(data)

# Apriori algorithm
frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# This generate association rule for columns
# comprises of antescends,consequences

rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

# Visualize the rules
import networkx as nx
import matplotlib.pyplot as plt

# Create directed graph from the rules
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents')

# Draw the graph
fig, ax = plt.subplots(figsize=(14, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)
plt.title("Association Rules Network", fontsize=15)
plt.show()

###################################################
# By using this data we can suggest to the customer which colour he/she
# should selec for the mobile 

*******************
*******5.) transaction_retail.csv****
'''
Problem Statement: - 
A retail store in India, has its transaction data, and it would like to know the buying pattern of the 
consumers in its locality, you have been assigned this task to provide the manager with rules 
on how the placement of products needs to be there in shelves so that it can improve the buying
patterns of consumes and increase customer footfall. 
5.) transaction_retail.csv

'''


retail=[]
#and store it into list
with open(r"D:\DATA SCIENCE\Machine_learning\ML ALGORITHM\aproiry\transactions_retail1.csv") as f:retail=f.read()
retail

retail=retail.split('\n')
#here we split the data from \n

retail_list=[i.split(',') for i in retail]
retail_list
#here we split the transactions from ,

all_retail=[j for i in retail_list for j in i]
all_retail
len(all_retail)
#here our all transactions_retail data is properly distribute/splited
#now lets count the values

#------------count 

from collections import Counter
item_freq=Counter(all_retail)
item_freq
# item_frequencies will be contain key and dictionary 
# we want to sort it into count frequencies 
# means it will show he count of item purchased
# let us sort the frequencies in ascending order

item_freq=sorted(item_freq.items(),key=lambda x:x[1])
item_freq

#now seperate the item and key in reverse format

item=list(reversed([i[0] for i in item_freq]))
item

freq=list(reversed([i[1] for i in item_freq]))
freq
#here we seperate the frequencies and items

#-----------now plot the graph

import matplotlib.pyplot as plt
plt.bar(height=freq[0:25],x=list(range(0,25)))
plt.xlabel('items')
plt.ylabel('count')
plt.show()

#now convert it into Data Frame

import pandas as pd
from mlxtend.frequent_patterns import association_rules,apriori
df=pd.DataFrame(pd.Series(retail_list))

#here we remove the last empty list
df=df.iloc[0:50000,:]
df

#now rename the column
df=df.rename(columns={0:'retail'})
df

#now join the values using *
x=df['retail'].str.join(sep='*')
x

#now apply one hot encoder on it
y=x.str.get_dummies(sep='*')


#now apply apriori algorithm
retail_freq=apriori(y,min_support=0.0075,max_len=4,use_colnames=True)
retail_freq

retail_freq.sort_values('support')

rules=association_rules(retail_freq,metric='lift',min_threshold=1)
rules.head()


*******************8