# Instagram Reach Analysis using Python

![instalogo](https://github.com/Priyaah13/priya.github.io/blob/main/instalogo.jpg)


## Introduction

Instagram is the most popular social media applications today. People use Instagram for their business, building a portfolio, blogging, and creating various kinds of content. This boomed content creators and the users. As IG keeps changing, it affects the reach of your posts that affects in the long run. So if a content creator wants to do well on Instagram in the long run, they have to look at the data of their Instagram reach. That is where the use of Data Science in social media comes in. If you want to learn how to use our Instagram data for the task of Instagram reach analysis, this article is for you. In this article, I will take you through Instagram Reach Analysis using Python, which will help content creators to understand how to adapt to the changes in Instagram in the long run.

## Instagram Reach Analysis using Python

Let’s start the task of analyzing the reach of my Instagram account by importing the necessary Python libraries and the dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv("Instagram.csv", encoding = 'latin1')
print(data.head())

## Check for Null values

data.isnull().sum()

## data types

data.info()

## Analyzing Instagram Reach: Distribution of impressions

plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.distplot(data['From Home'])
plt.show()

## Distribution of Impressions From Hashtags

plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of impressions from Hashtags")  # Corrected to plt.title()
sns.histplot(data['From Hashtags'], kde=True)
plt.show()

## Distribution of Impressions From Explore

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(data['From Explore'])
plt.show()

Home Impressions: It's challenging to reach all followers daily, suggesting limitations in organic reach.
Hashtag Impressions: While hashtags can expand reach to new users, they are not a guaranteed solution for all posts.
Explore Impressions: Instagram's recommendation system doesn't seem to prioritize the posts significantly, with limited exposure from Explore.
Overall, the analysis highlights the complexities of organic reach on Instagram, suggesting that a combination of strategies, including consistent posting, engaging content, and strategic hashtag use, may be necessary to maximize visibility.

## Percentage of impressions

home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()

The above donut plot shows that almost 50 per cent of the reach is from my followers, 38.1 per cent is from hashtags, 9.14 per cent is from the explore section, and 3.01 per cent is from other sources.

## Analysing Content Here. the dataset has two columns, namely caption and hashtags, which will help to understand the kind of content posted on Instagram. Createte a wordcloud of the caption column to look at the most used words in the caption

text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

## The most used hashtags in Instagram posts

text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

## Analysing Relationships. Understanding how the Instagram algorithm works. Relationship between the number of likes and the number of impressions on my Instagram posts.

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()

## Now let’s see the relationship between the number of comments and the number of impressions on Instagram posts:

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()

## Relationship Between Shares and Total Impressions

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()

The analysis reveals the following relationships between Instagram metrics and reach:

Likes and Reach: A strong positive correlation exists, indicating that more likes generally lead to increased reach.
Comments and Reach: No significant correlation, suggesting that comments alone don't significantly impact reach.
Shares and Reach: A positive correlation exists, but it's less pronounced than the relationship between likes and reach.
Saves and Reach: A strong positive correlation exists, suggesting that saves are a strong indicator of potential reach.
In conclusion, while likes and saves are crucial for increasing reach, comments and shares, though important, have a less direct impact.

## Data correlation 

# Select only numeric columns
numeric_data = data.select_dtypes(include=[float, int])

# Calculate the correlation matrix
correlation = numeric_data.corr()

# Display correlation with the 'Impressions' column
print(correlation["Impressions"].sort_values(ascending=False))

Based on the analysis, it's clear that likes and saves are significant factors in boosting Instagram reach. While shares can also contribute to increased visibility, their impact is less pronounced compared to likes and saves.

## Analyzing Conversion Rate

In Instagram, conversation rate means how many followers you are getting from the number of profile visits from a post. The formula that you can use to calculate conversion rate is (Follows/Profile Visits) * 100. Now let’s have a look at the conversation rate of my Instagram account:

conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)

## A 41% conversation rate on Instagram is indeed exceptional. This suggests that a significant portion of your audience is actively engaging with your content, whether it's through likes, comments, or direct messages.

## Relationship between the total profile visits and the number of followers gained from all profile visits:

figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()

## Instagram Reach Prediction Model

Now in this section, I will train a machine learning model to predict the reach of an Instagram post. Let’s split the data into training and test sets before training the model:

x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

##  train a machine learning model to predict the reach of an Instagram post using Python:

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

## 0.8227437065296919

## Now let’s predict the reach of an Instagram post by giving inputs to the machine learning model:

# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)

## array([11907.32299294])

## Summary:

## Follower Reach Dominance: Nearly 50% of the total reach originates from direct follower engagement, highlighting the importance of nurturing a loyal following.
## Hashtag Impact: While hashtags contribute significantly (38.1%) to expanding reach, their effectiveness can vary. Strategic hashtag selection is crucial.
## Limited Explore Page Exposure: The Explore page, a potential source of significant organic reach, contributes only 9.14%. This underscores the need for creative content and consistent posting to attract algorithm-driven recommendations.
## The Power of Likes and Saves: A strong positive correlation between likes and saves with reach suggests that user engagement is a key driver of visibility.
## Comments and Shares: A Secondary Role: While comments and shares can contribute to reach, their impact is less pronounced compared to likes and saves.
## High Conversation Rate: A 41% conversation rate indicates a highly engaged audience, which is a valuable asset for content creators.

## In model training:
Score (0.82): This indicates the R-squared value, which measures the proportion of variance in the actual reach (ytest) explained by the model's predictions. A value of 0.82 suggests a relatively good fit, meaning the model explains 82% of the variance in actual reach.

## Prediction:

Prediction (11907.32): The model predicts that a post with these specific engagement metrics could potentially reach approximately 11,907 users.

