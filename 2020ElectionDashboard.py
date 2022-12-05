#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basics and Plotting
import html as html
import pandas as pd
import numpy as np
# import scipy as scp
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# import seaborn as sns
from itertools import chain, combinations
import plotly.express as px
import plotly.graph_objects as go
import dash
#from dash import dcc
from dash import Dash, html, dcc
#import dash_html_components as html
from dash.dependencies import Input, Output

# Sklearn Models
# import sklearn.linear_model as skl_lm
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.linear_model import Lasso, Ridge
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score, cross_validate
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.preprocessing import scale
# from sklearn.decomposition import PCA
# from sklearn.cross_decomposition import PLSRegression
#
# # Alternative models
# import statsmodels.api as sm
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import statsmodels.formula.api as smf


# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/jprichmond20/2020ElectionStudy/main/anes_timeseries_2020_csv_20220210.csv")


# In[3]:


df = df[["V201004", "V201005", "V201008",
                    "V201024", "V202110x", "V201030", "V201651",
                    "V201645", "V201628", "V201627", "V201626",
                    "V201622", "V201620", "V201607", "V201606",
                    "V201602", "V201600", "V201601", "V201594",
                    "V201589", "V201575", "V201565x", "V201550",
                    "V201549x", "V201540", "V201531",
                    "V201530y", "V201515", "V201511x", "V201508",
                    "V201507x", "V201462", "V201453", "V201435",
                    "V201433", "V201432x", "V201426x", "V201417",
                    "V201416", "V201411x", "V201393", "V201379",
                    "V201377", "V201366", "V201364", "V201359x",
                    "V201335", "V201244", "V201242", "V201241",
                    "V201240", "V201239", "V201237", "V201231x",
                    "V201232", "V201220", "V201216", "V201156",
                    "V201157", "V201122", "V201123", "V201121",
                    "V201114", "V201100", "V201103", "V201036",
                   "V201018"]]


# In[4]:


df = df.iloc[:,[4,64,63,49,2,17,56,38,42,28,50,15,9,11,6,58,65,8,27]]


# In[5]:


df = df.rename(columns = {'V202110x': 'VFor','V201103': 'VFor2016' , 'V201100': 'Likely2V', 'V201241':'PBetterImmigration', 'V201008':'R2V', 'V201601':'SexualOrientation', 'V201216':'CareWhoWins', 'V201416':'GayMarriage', 'V201377':'TrustMedia', 'V201511x':'LevelOfEducation', 'V201240':'PBetterHealthcare', 'V201602':'JustifiedPViolence', 'V201627':'SelfCensor', 'V201622':'ConcernedPay4Healthcare', 'V201651':'SatisfiedWLife', 'V201157':'FThermometerR', 'V201036':'CandidatePref', 'V201628':'GunsOwned', 'V201515':'DiplomaOrGED'})


# In[6]:


df.head()


# In[7]:


df['VFor'].value_counts()


# In[12]:


df2 = df.copy()
f1 = df2['VFor']==1
f2 = df2['VFor']==2
df2['VFor'].where(f1|f2, inplace = True)
df2.head()


# In[13]:


df2 = df2.dropna()
df2.head()


# In[14]:


df2.head()


# In[17]:


#1 = Biden 2 = Trump
df2.replace(1.0, "Biden", inplace = True)
df2.replace(2.0, "Trump", inplace = True)


# In[18]:


fig1 = px.bar(df2, x=df2['VFor'].unique(),               y=df2.groupby('VFor')['VFor'].agg('count'),               labels={"x":"Candidate", "y":"# Of Voters"},               color_continuous_scale=px.colors.sequential.Hot,               title="Votes For Candidate")
fig1.update_xaxes(type='category')
fig1.update_layout(title=dict(x=0.5),                  margin=dict(l=20,r=20,t=60,b=20),                  paper_bgcolor="#D3D3D3")


# In[19]:


df.head()


# In[20]:


df3 = df.copy()
df3.head()


# In[21]:


f1 = df3['VFor']==1
f2 = df3['VFor']==2
df3['VFor'].where(f1|f2, inplace = True)
df3 = df3.dropna()
df3.head()


# In[ ]:





# In[22]:


df3['VFor'] = df['VFor'].map({1.0:"Biden", 2.0:"Trump"})
df3.head()


# In[23]:


indexNames = df3[ df3['SatisfiedWLife'] < 0 ].index

# Delete these row indexes from dataFrame
df3.drop(indexNames , inplace=True)
indexNames2 = df3[ df3['PBetterImmigration'] < 0 ].index
df3.drop(indexNames2 , inplace=True)
df3.head()


# In[27]:


fig2 = px.scatter(df3, x=df3['PBetterImmigration'], \
              y=df3['VFor'], color=df3['FThermometerR'], size=df3['SatisfiedWLife'], opacity=0.01)
fig2.update_xaxes(type='category', categoryorder='category ascending')


# In[28]:


df['GunsOwned'].value_counts()


# In[36]:


df3 = df3.sort_values(by=['GunsOwned'], ascending=True)


# In[38]:


indexNames = df3[ df3['TrustMedia'] < 0 ].index
# Delete these row indexes from dataFrame
df3.drop(indexNames , inplace=True)
df3.head()


# In[39]:

fig3 = px.scatter(df3, x=df3['GunsOwned'], \
              y=df3['TrustMedia'], \
                  color=df3['VFor'], \
              labels={"x":"Candidate", "y":"# Of Voters"}, \
              color_continuous_scale=px.colors.sequential.Hot, \
              title="Votes For Candidate", opacity=0.15)
fig3.update_xaxes(type='category', categoryorder='array', categoryarray=[-9,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 23, 24, 25, 29, 30, 36, 40, 44, 50, 60, 78, 99])#, axis={'categoryorder':'category ascending'})
fig3.update_yaxes(type='category', categoryorder='category ascending')
fig3.update_layout(title=dict(x=0.5),\
                  margin=dict(l=20,r=20,t=60,b=20),\
                  paper_bgcolor="#D3D3D3")
# fig3 = px.scatter(df3, x=df3['GunsOwned'],               y=df3['TrustMedia'],                   color=df3['VFor'],               labels={"x":"Candidate", "y":"# Of Voters"},               color_continuous_scale=px.colors.sequential.Hot,               title="Votes For Candidate", opacity=0.15)
# fig3.update_xaxes(type='category')
# fig3.update_layout(title=dict(x=0.5),                  margin=dict(l=20,r=20,t=60,b=20),                  paper_bgcolor="#D3D3D3")




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True


# children=[html.Div(children='yearID',\
#                                 style={'fontSize':"20px"},\
#                                 className='menu-title-bar'),\
#                                 dcc.Dropdown(id='year-selection',\
#                                 options=[{'label':yearID,'value':yearID}\
#                                 for yearID in baseball2.yearID.unique()],\
#                                 value='2016',\
#                                 clearable=False,\
#                                 searchable=False,\
#                                 className='dropdown',\
#                                 style={'fontSize':"20px",'textAlign':'center'})],\
#              className='menu'),
app.layout = html.Div(children=[    html.Div(children=[
        html.H1(children="2020 Election Survey", style={'textAlign':'center'},className="header-text"),
        html.H2(children="JP Richmond",style={'fontSize':"30px",'textAlign':'center'},className="names-header"),

    html.Div(children=[
        html.Div(
            children = dcc.Graph(
                    id = 'Votes',
                    figure = fig1,
                  #  config={"displayModeBar": False},
                ),
                style={'width': '50%', 'display': 'inline-block'},
            ),
                html.Div(\
                children = dcc.Graph(
                    id = '4Var',
                    figure = fig2,
                    #config={"displayModeBar": False},
                ),
                style={'width': '50%', 'display': 'inline-block'},
            ),
                html.Div(
                children = dcc.Graph(
                    id = 'GunsVsMedia',
                    figure = fig3,
                    #config={"displayModeBar": False},
                ),
                         style={'width': '100%', 'display': 'inline-block'})

    ])])])
#])


# In[ ]:


# @app.callback(Output("Votes","figure"),Input("years","value"))
# def update_charts(years):
#     filtered_data = df[df["yearID"] == years]
#     barHR = px.bar(filtered_data,x="teamID",y="HR",title="Home Runs by Team",color_continuous_scale=px.colors.sequential.Plasma)
#     barHR.update_layout(xaxis_tickangle=20,title=dict(x=0.5),xaxis_tickfont=dict(size=8),yaxis_tickfont=dict(size=8),paper_bgcolor="LightSteelBlue",margin=dict(l=30,r=20,t=50,b=20))
#     return barHR
# @app.callback(Output("4Var","figure"),Input("years","value"))
# def update_charts(years):
#     filtered_data = df[df["yearID"] == years]
#     barHRLg = px.bar(filtered_data,x="lgID",y="HR",title="Home Runs by League",color_continuous_scale=px.colors.sequential.Plasma)
#     barHRLg.update_layout(xaxis_tickangle=20,title=dict(x=0.5),xaxis_tickfont=dict(size=8),yaxis_tickfont=dict(size=8),paper_bgcolor="LightSteelBlue",margin=dict(l=30,r=20,t=50,b=20))
#     return barHRLg
# @app.callback(Output("GunsVsMedia","figure"),Input("years","value"))
# def update_charts(years):
#     filtered_data = df[df["yearID"] == years]
#     barSO = px.bar(filtered_data,x="teamID",y="SO",title="Strikeouts by Team",color_continuous_scale=px.colors.sequential.Plasma)
#     barSO.update_layout(xaxis_tickangle=20,title=dict(x=0.5),xaxis_tickfont=dict(size=8),yaxis_tickfont=dict(size=8),paper_bgcolor="LightSteelBlue",margin=dict(l=30,r=20,t=50,b=20))
#     return barSO


# In[ ]:
if __name__ == '__main__':
     app.run_server(host='0.0.0.0', port=8080, debug=True, use_reloader=False)




# In[ ]:





# In[ ]:





# In[ ]:




