

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from threading import RLock


st.header(":blue[Analyzing Data from Video Game Sales]")

st.write("The objective of the project is to carryout an analysis on database form global online store Ice to identify patterns that determine whether a game succeeds or not")    

df_vehicles = pd.read_csv('games.csv')

st.write(df_game.sample(10))


df_game.columns = df_game.columns.str.lower() #converting the columns to lower case.
df_game['user_score'] = pd.to_numeric(df_game['user_score'], errors='coerce')

df_game['genre'] = df_game['genre'].fillna(df_game['genre'].mode()[0])  #filled the nan value in genre with mode.

df_game['name'] = df_game['name'].fillna('unknown') #filled the nan value in name with unknown
#This is because it not clear what values the name and the genre represent.

df_game['user_score'] = df_game['user_score'].fillna(df_game.groupby(['genre', 'platform'])
                                                     ['user_score'].transform(lambda x:x.fillna(x.median())))
#replacing missing critic scores with the median score ensures that analyses remain unbiased.

# Convert 'Year_of_Release' to datetime format to allow for easy analysis
df_game['year_of_release'] = pd.to_datetime(df_game['year_of_release'], format='%Y')


# Fill NaNs with the mode of 'Year_of_Release' since it is hard to identify pattern with the name. 
df_game['year_of_release'] = df_game['year_of_release'].fillna(df_game['year_of_release'].mode()[0]) 


df_game['critic_score'] = df_game['critic_score'].fillna(df_game.groupby(['platform','genre'])
                                                         ['critic_score'].transform(lambda x:x.fillna(x.median())))
 #replacing missing critic scores with the median score ensures that analyses remain unbiased.

df_game['user_score'] = df_game['user_score'].fillna(df_game.groupby(['genre'])
                                                     ['user_score'].transform(lambda x:x.fillna(x.median())))
df_game['critic_score'] = df_game['critic_score'].fillna(df_game.groupby(['genre'])
                                                         ['critic_score'].transform(lambda x:x.fillna(x.median())))
# filling NaNs with a placeholder such as 'Not Rated' is standard practice.
df_game['rating'] = df_game['rating'].fillna('Not Rated')

df_game['total_sale'] = df_game[['eu_sales','jp_sales','na_sales', 'other_sales']].sum(axis=1)



_lock = RLock()
st.header("Dertermine the games released in different years and their significant")
#extract year from 'Year' column
df_game['year_release'] = df_game['year_of_release'].dt.year
# Count games released per year
games_per_year = df_game.groupby('year_release')

with _lock:
    fig_1 = games_per_year.size().plot(kind='bar')
    plt.title("games released per year")
    plt.xlabel("year")
    plt.ylabel("number of games")
    plt.show()
    st.pyplot (fig_1)

st.write (
"The bar chart suggest that the highest sales occured in 2008, 2009, and 2010, respectively."

"It can be said that the their mojor sales occured between 2002 and 2011."

"That there is significant difference between the observed and expected distributions, indicating that the number of games released each year varies ignificantly."
"The data for every period significant is not signicant and can be dropped"
)


st.header("Analyze Sales Trends Over Time for Each Platform:")
# Analyzing how sales varied from platform to platform using Bar Chart
_lock = RLock()
with _lock:
    fig_2 = df_game.groupby('platform')['total_sale'].sum().plot(kind='bar')
    plt.title("sales by platform")
    plt.xlabel("platform")
    plt.ylabel("sales")
    plt.show()
    st.pyplot (fig_2)
st.write(
"The bar chart indicates that there is significant variation in the sale per platform."
"The highest sale occured in PS2 platform, x360, Wii, PS3, DS, and Ps, respectively."
"No observed sales on the GG, 3D0, and TG16.")


st.header("Visualing platforms Sales")
# Grouping by 'Year' and 'Platform' and sum the 'Sales'
year_platform_sales = df_game.groupby(['year_release', 'platform'])['total_sale'].sum().reset_index()

# Creating a pivot table for better visualization
year_platform_sales_pivot = year_platform_sales.pivot(index='year_release', columns='platform', values='total_sale')

# Creating a figure and axis
_lock = RLock()
cmap = plt.get_cmap("tab20")
cmap1 = plt.get_cmap("tab20b")
with _lock:
    plt.figure(figsize=(14, 8))

# Plot sales trends for each platform
    for i, platform in enumerate(year_platform_sales_pivot.columns):
        if i < 20:
            fig_3 = plt.plot(year_platform_sales_pivot.index, year_platform_sales_pivot[platform], marker='o', label=platform, color=cmap(i))
        else:
            fig_3 = plt.plot(year_platform_sales_pivot.index, year_platform_sales_pivot[platform], marker='o', label=platform, color=cmap1(i-20))

    plt.title('Yearly Sales Trends by Platform')
    plt.xlabel('Year')
    plt.ylabel('Total Sales')

    plt.legend(title='platform', loc="upper left", bbox_to_anchor=(1,1))
    plt.tight_layout()

    plt.show()
    st.pyplot (fig_3)





st.header("Visualing 10 platforms with the greatest total sales Top")

total_sales_platform = df_game.groupby('platform')['total_sale'].sum().sort_values(ascending=False) # Calculate Total Sales by Platform

top_platforms_sales = total_sales_platform.head(10).index # 10 Sale by Platform



# Building a Distribution of Sales by Year for the Top 10 Platforms

df_top_platforms_sales = df_game[df_game['platform'].isin(top_platforms_sales)] # Filter the DataFrame for the top 10 platforms only.

top_platforms_yearly_sales = df_top_platforms_sales.groupby(['year_release', 'platform'])['total_sale'].sum().unstack()

_lock = RLock()
#Visualize the distribution with a line plot
with _lock:
    plt.figure(figsize=(10, 6))
    for platform in top_platforms_yearly_sales.columns:
        fig_4 = plt.plot(top_platforms_yearly_sales.index, top_platforms_yearly_sales[platform], marker='o', label=platform)
    plt.title('Yearly Sales Distribution for Top Platforms')
    plt.xlabel('Year')
    plt.ylabel('Total Sales')
    plt.legend(title='Platform')
    plt.grid(True)
    plt.show()
    st.pyplot (fig_4)

st.write(
'The chart indicates that there is significant variation in the sale per platform overtime.'
'The platforms that used to be popular but now have zero sales are PS2, PS, DS, and Wii.'

'From the plot, it roughly takes about 6 to 8 years for new platforms to appear and old ones to fade.'

'We can take data from 1995 to 2016 to build a model for 2017.'

'The platforms that were leading in sales are PS, PS2, Wii, PS3, and X360; but they are almost fading out.'
 
'The PS, PS2, and DS, despite doing will ealier have completely fizzle out.'

'No potentially profitable platforms; Although, PS4, Xone, and 3DS were leading, they were alreading trending downward.'
)


st.header("Box plot for the global sales of all games, broken down by platform")
df_selected_game = df_game[df_game['year_release'] > 1995]  #Selecting data from 1995 to 2016
total_sale_quantile_99 =  df_selected_game.total_sale.quantile(q=0.99) #removing the outliers.
df_selected_game_99 = df_selected_game.loc[df_selected_game.total_sale <= total_sale_quantile_99]

_lock = RLock()
# Plotting the box plot.

with _lock:
    plt.figure(figsize=(18, 25))
    fig_5 = sns.boxplot(x='platform', y='total_sale', data=df_selected_game_99)
    plt.xticks(rotation=45)
    plt.title('Global Sales Distribution by Platform')
    plt.xlabel('Platform')
    plt.ylabel('Global Sales (millions)')
    plt.show()
    st.pyplot(fig_5)

st.write(
'The box plots showed that there are significant variations in the sales per platform.'

'Majority of the platforms such as Wii, X360, Ps3,PS4, Xone have outliers that are skewed to the right (indicating high sales).'

'Howerver, platforms like GEN and GB are evenly distributed.'

'There is significant variation in the mean values of each platform.'
)

agree = st.checkbox("Acceped")

if agree:
    st.write("Great, moving on o next step!")
