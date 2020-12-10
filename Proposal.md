# Final Project Proposal

Group Members: Jiayi Weng, Xinyue Chen, Xuanyi Li, Yeju Zhou

## Overview

 is a dataset that contains various information extracted from Stack Overflow posts and users, including the user profiles, questions, answers, ratings, votes, etc. This dataset brings about interesting objective questions for us to discover throughout this project and enables us to conduct a variety of exploratory analysis and model development that is backed by massive amounts of data, which will be discussed later. This project aims to gain insights that help us better understand the users and discussion topics active in the Stack Overflow community and make the community more efficient and better connected. The project will follow the **interactive visualization and application track**. 

## Problem & Solution

1. How do users that answered similar questions or were active under similar topics relate with each other? We will apply state-of-the-art NLP models to extract the keywords regarding both the users and question-answer pairs. Based on the statistics of the results, we will dig into the user network in Stack Overflow to discover the similarities between users. Also, we expect to locate small communities and coteries among users who have answered similar questions.
2. How do we recommend users for users to connect and questions for users to answer? We will recommend topics/questions to a user given his/her posting history; we will recommend users with similar interest and technical focus to a user given their posting history. We will also recommend the potential users who are more likely to answer a specific question given his/her answering history; and recommend related or similar questions that have already been answered when a user posts a new question.
3. We hope to find out what kind of answers tend to earn a higher rating. And what kind of users are likely to receive more votes and gain greater influence? We will analyze the best rated answers and the most voted users from a statistical perspective. We will try to uncover and demonstrate the patterns and popularities of different technical topics, question tags, answer styles and expertise domains that received a higher rating and a greater impact. 

## Visualization Design

1. We extract the keywords related to users, questions, answers, etc. We will design a heatmap like visualization to emphasize the popular topics, top rated posts, etc. Tooltips will give example posts that contain or focus on a word.
2. User-to-user relationships will be represented and visualized in a tree-like or network-like graph with circular nodes. Users with similar interests and similar technical focuses will gather closer to each other and be connected with lines. Selecting a user will pop out a profile-like view that shows selected information of the user. Moreover, there will be options that allow the application to recommend users and question given a sample user or sample question in real time.
3. We will quantify and visualize the statistics in an iteratively exploratory manner. We will first discover the data distribution and show the distribution with simple visualization designs like bar charts and line charts. Then we will explore the data and try to identify latent patterns and relationships between factors and variables. Later, we will show the patterns and relationships that are interesting and insightful with appropriate visual designs depending on the data type, data distribution, question we ask, etc.