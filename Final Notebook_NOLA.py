#!/usr/bin/env python
# coding: utf-8

# # NOLA Hurricanes
# ## Marlee Walls, Helen Charbonnet, Megan Zhang, Mary Green

# ### Predictive Flooding Model
# #### Imports

# In[119]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Reading in and manipulating data

# In[120]:


# Read in data and display first 5 rows
features = pd.read_csv('78_Accurate.csv')
# Use numbers to correspond to dates from 2007-2020
features.head(5)


# In[121]:


# printing feature shape
print('The shape of our features is:', features.shape)


# In[122]:


# Descriptive statistics for each column
features.describe()
# drop the rows with NaN
features.dropna()


# In[123]:


# Use numpy to convert to arrays
import numpy as np# Labels are the values we want to predict
labels = np.array(features['Gage Height'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Gage Height', axis = 1)# Saving feature names for later use
feature_list = list(features.columns)# Convert to numpy array
features = np.array(features)


# #### Training and Testing Sets

# In[124]:


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[125]:


# print the shapes for each
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# ##### Average Baseline Error

# In[126]:


# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('Average Tide')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)

print('Average baseline error: ', 
round(np.mean(baseline_errors), 2))


# In[127]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 12 decision trees
rf = RandomForestRegressor(n_estimators = 12, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);


# ##### Mean Absolute Error

# In[128]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# ##### Final Accuracy

# In[129]:


# replacing 0 with averages
i = 0
while i < 1096:
    if test_labels[i] == 0.:
        test_labels[i] = 0.8785438556
    i+=1
    
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
print(test_labels)
np.trim_zeros(test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print(accuracy)
print('Accuracy:', round(accuracy, 2), '%.')


# #### Checking Variable Importance

# In[130]:


# Checking variable importance
# Instantiate random forest and train on new features
from sklearn.ensemble import RandomForestRegressor
rf_exp = RandomForestRegressor(n_estimators= 100, random_state=10)
rf_exp.fit(train_features, train_labels)

# Get numerical feature importances
importances = list(rf_exp.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# ### Feature Graphs

# #### Precipitation

# In[131]:


#read data from csv
precip_data = pd.read_csv('Precipitation_test.csv', usecols=['Date','Precipitation'], parse_dates=['Date'])
#set date as index
precip_data.set_index('Date',inplace=True)

#set ggplot style
plt.style.use('ggplot')

#plot data
p_fig, p_ax = plt.subplots(figsize=(25,10))

# data.plot(kind='bar', ax=ax)
p_ax.bar(precip_data.index, precip_data['Precipitation'])

#plt.xticks(rotation=70)
p_ax.tick_params(axis="x", labelsize=20)
p_ax.tick_params(axis="y", labelsize=20)

p_ax.set_title('Daily Precipitation 2007-2020 NOLA', fontsize=30)
p_ax.set_ylabel('Inches', fontsize=30)
p_ax.set_xlabel('Date', fontsize=30)


# #### Tides

# In[132]:


### reading in datasheet
tide_data = pd.read_excel('./Feb_Tides.xlsx')
tide_data

### plot of just high tides
t_fig, t_ax = plt.subplots()
t_ax.set_xlabel("Date Recorded")

t_ax.set_ylabel("High Tide")
t_ax.set_title("Hide Tide Levels Of Lake Pontchartrain 2007-2020", fontsize=20)

plt.plot(tide_data["Date"],  # x values
   tide_data["High Tide"], # y values
       )
plt.show()


# In[133]:


### plot of just low tides
tt_fig, tt_ax = plt.subplots()
tt_ax.set_xlabel("Date Recorded")

tt_ax.set_ylabel("Low Tide")
tt_ax.set_title("Low Tide Levels Of Lake Pontchartrain 2007-2020", fontsize=20)

plt.plot(tide_data["Date"],  # x values
    tide_data["Low Tide"], # y values
        )
plt.show()


# In[134]:


### plot of avg tide 
ttt_fig, ttt_ax = plt.subplots()
ttt_ax.set_xlabel("Date Recorded")

ttt_ax.set_ylabel("Average Tide")
ttt_ax.set_title("Average Tide Levels Of Lake Pontchartrain 2007-2020", fontsize=20)

plt.plot(tide_data["Date"],  # x values
    tide_data["Average Tide"], # y values
        )
plt.show()


# In[135]:


### plot containing all three for comparison
tttt_fig, tttt_ax = plt.subplots()
tttt_ax.set_xlabel("Date Recorded")

tttt_ax.set_ylabel("Tide Levels (feet)")
tttt_ax.set_title("February Tide Fluctuation Of Lake Pontchartrain", fontsize=20)

plt.plot(tide_data["Date"], tide_data["High Tide"], "b--", 
         tide_data["Date"], tide_data["Average Tide"], "g--", 
         tide_data["Date"], tide_data["Low Tide"], "r--")
plt.show()


# #### Hurricane Season

# In[136]:


#read data from csv
data = pd.read_csv('Hurricane Season Year.csv')

#plot data
plt.figure(figsize=(20,10))
plt.suptitle('Hurricane Season Impact by Month', fontsize=20)
plt.xlabel('Month', fontsize=18)
plt.ylabel('Hurricane Season', fontsize=16)
plt.plot(data.Month, data.Season)


# #### Sea Level Temperature

# In[137]:


#read data from csv
data = pd.read_csv('Sea Level Temp Year - Sheet1-1.csv')

#plot data
plt.figure(figsize=(20,10))
plt.suptitle('Sea Level Temperature by Month', fontsize=20)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Degrees', fontsize=16)
plt.plot(data.Date, data.Sea_Level_Temp)


# #### Historical Hurricanes

# In[138]:


import matplotlib.pyplot as plt
#read data from csv
precip_data = pd.read_csv('Historical Hurricanes_test.csv', usecols=['Date','Historical_Hurricanes'], parse_dates=['Date'])
#set date as index
precip_data.set_index('Date',inplace=True)

#set ggplot style
plt.style.use('ggplot')

#plot data
p_fig, p_ax = plt.subplots(figsize=(25,10))

# data.plot(kind='bar', ax=ax)
p_ax.bar(precip_data.index, precip_data['Historical_Hurricanes'])

#set x axis limits
plt.xlim(xmin='1/1/08')
plt.xlim(xmax='12/31/14')

p_ax.tick_params(axis="x", labelsize=20)
p_ax.tick_params(axis="y", labelsize=20)

p_ax.set_title('Historical Hurricanes 2007-2020 NOLA', fontsize=30)
p_ax.set_ylabel('Hurricane Magnitude', fontsize=30)
p_ax.set_xlabel('Date', fontsize=30)


# ### Location Based Graphs

# #### Location of Levees

# In[139]:


#Imports 
import pandas as pd
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon 

import hvplot.xarray
import holoviews as hv
hv.extension('bokeh')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[140]:


### DATA 1: Location of Levees 

data1 = pd.read_csv('./Segment.csv')
crs = {'int':'epsg:4326'}
data1

selection1 = data1[['BEGIN LONGITUDE', 'END LONGITUDE']]
selection2 = data1[['BEGIN LATITUDE', 'END LATITUDE']]

### creating bounds for maps
BBox = ((data1["BEGIN LONGITUDE"].min(), data1["BEGIN LONGITUDE"].max(), data1["BEGIN LATITUDE"].min(), data1["BEGIN LATITUDE"].max()))
BBox

### import maps with the same bounds
nola_map = plt.imread('./map.png')

### scatter plot on the map
fig, ax = plt.subplots(figsize=(8,7))
ax.scatter(data1["BEGIN LONGITUDE"], data1["BEGIN LATITUDE"], zorder=1, alpha=0.99, c='b', s=10)
ax.scatter(data1["END LONGITUDE"], data1["END LATITUDE"], zorder=1, alpha=0.99, c='r', s=10)
ax.set_title('Levees in Orleans Parish')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

ax.imshow(nola_map, zorder=0, extent=BBox, aspect='equal')


# #### Elevation of Levees

# In[141]:


### New Orleans East Bank
## reading in data set 
dataEAST = pd.read_csv('./system_4405000556_profile.csv')
dataEAST

 ## EAST
plt.figure(figsize=(20,10))
plt.plot(dataEAST[" m"],  # x values
    dataEAST[" z"], # y values
)
plt.xlabel("Distance along Levee (ft)", fontsize=20)

plt.ylabel("Elevation along Levee (ft)", fontsize=20)
plt.title("Elevation of New Orleans East Bank", fontsize=20)


# In[142]:


### New Orleans West Bank
## reading in data set 
dataWEST = pd.read_csv('./system_4405000557_profile-1.csv')
dataWEST

## WEST
plt.figure(figsize=(20,10))
plt.plot(dataWEST[" m"],  # x values
    dataWEST[" z"], # y values
)
plt.xlabel("Distance along Levee (ft)", fontsize=20)

plt.ylabel("Elevation along Levee (ft)", fontsize=20)
plt.title("Elevation of New Orleans West Bank", fontsize=20)


# In[143]:


### Mississippi River East Bank 
## reading in data set 
data3 = pd.read_csv('./system_4405000501_profile-1.csv')
data3

## MS EAST
plt.figure(figsize=(20,10))
plt.plot(data3[" m"],  # x values
    data3[" z"], # y values
)
plt.xlabel("Distance along Levee (ft)", fontsize=20)

plt.ylabel("Elevation along Levee (ft)", fontsize=20)
plt.title("Elevation of Mississippir River East Bank", fontsize=20)


# #### Elevation Data Map

# In[144]:


### DATA 3: DEM elevation DATA 

### new imports 
get_ipython().system('pip install pygeotools')
from osgeo import gdal
import seaborn as sns
from pygeotools.lib import iolib, warplib, geolib, timelib, malib

### importing data and analyzing a smaller section 
dem_ne = gdal.Open('./dem_2909007ne.dem')
dem_arr1 = np.array(dem_ne.GetRasterBand(1).ReadAsArray())
im1 = plt.imshow(dem_arr1, vmin=-4, vmax=12)
im1.cmap.set_under('yellow')
im1.cmap.set_over('cyan')

### anaylzing larger area

get_ipython().system('wget https://www.ngdc.noaa.gov/thredds/fileServer/regional/new_orleans_13_mhw_2010.nc')

dem_data = gdal.Open('./new_orleans_13_mhw_2010.nc')
dem_array = np.array(dem_data.GetRasterBand(1).ReadAsArray())
print(dem_array)
print(dem_array.min(), dem_array.max())
print(dem_array.shape)
im = plt.imshow(dem_array, vmin=-2, vmax=14) ## play around with range here


im.cmap.set_under('yellow')
im.cmap.set_over('cyan')


# In[ ]:





# In[ ]:




