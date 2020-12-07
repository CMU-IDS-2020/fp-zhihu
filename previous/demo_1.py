import altair as alt
import numpy as np
import os
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import collections
from collections import Counter
from google.cloud import bigquery
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# global variables definition
MAX_WIDTH = 700

# update the secret to env
# this credential file should not commit to github
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'ids-final-876546687c97.json'

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "stackoverflow" dataset
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10 ** 11)


def serialization(id):
    if not isinstance(id, list):
        id = [id]
    if isinstance(id[0], str):
        id = ['`' + i + '`' for i in id]
    else:
        id = [str(i) for i in id]
    return '(' + ','.join(id) + ')'


@st.cache
def get_table(table_name, max_results=10000):
    # Construct a reference to the "posts_answers" table
    answers_table_ref = dataset_ref.table(table_name)

    # API request - fetch the table
    answers_table = client.get_table(answers_table_ref)

    # Preview the first five lines of the "posts_answers" table
    return client.list_rows(answers_table, max_results=max_results).to_dataframe()


@st.cache
def get_all_users(location=None, user_id=None, reputation_thres=None):

    if location:
        questions_query = f"WHERE location IN {serialization(location)}"
    elif user_id:
        questions_query = f"WHERE id IN {serialization(user_id)}"
    elif reputation_thres:
        questions_query = f"WHERE reputation >= {reputation_thres}"
    else:
        questions_query = ""
    questions_query = f"SELECT id, display_name, reputation, up_votes, down_votes FROM `bigquery-public-data.stackoverflow.users` " + questions_query

    # Set up the query (cancel the query if it would use too much of
    # your quota, with the limit set to 1 GB)
    questions_query_job = client.query(questions_query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame
    questions_results = questions_query_job.to_dataframe()

    return questions_results


@st.cache
def get_all_tags():
    questions_query = f"SELECT id, tag_name FROM `bigquery-public-data.stackoverflow.tags`"

    # Set up the query (cancel the query if it would use too much of
    # your quota, with the limit set to 1 GB)
    questions_query_job = client.query(questions_query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame
    questions_results = questions_query_job.to_dataframe()

    # return a list of tags
    return list(questions_results.iloc[:, 1])


@st.cache
def get_questions_by_user(user_id):
    questions_query = f"SELECT * FROM `bigquery-public-data.stackoverflow.posts_questions` \
                        WHERE owner_user_id IN {serialization(user_id)}"
    # Set up the query (cancel the query if it would use too much of
    # your quota, with the limit set to 1 GB)
    questions_query_job = client.query(questions_query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame
    questions_results = questions_query_job.to_dataframe()

    return questions_results


@st.cache
def get_answered_questions_by_user(user_id):
    questions_query = f"SELECT a.id, a.body, q.tags, a.owner_user_id AS uid, q.id AS qid \
                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q \
                INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a \
                    ON q.id = a.parent_id \
                WHERE a.owner_user_id IN {serialization(user_id)}"
    # Set up the query (cancel the query if it would use too much of
    # your quota, with the limit set to 1 GB)
    questions_query_job = client.query(questions_query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame
    questions_results = questions_query_job.to_dataframe()

    # Preview results
    return questions_results


@st.cache
def get_tag_by_question(qid):
    questions_query = f"SELECT tags \
                FROM `bigquery-public-data.stackoverflow.posts_questions` \
                WHERE id IN {serialization(qid)}"
    # Set up the query (cancel the query if it would use too much of
    # your quota, with the limit set to 1 GB)
    questions_query_job = client.query(questions_query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame
    questions_results = questions_query_job.to_dataframe()

    res = questions_results.iloc[0, 0].split('|')
    return res


@st.cache
def get_question_tag_by_user(user_id):
    tag_query = f"SELECT tags \
        FROM `bigquery-public-data.stackoverflow.posts_questions` \
        WHERE owner_user_id IN {serialization(user_id)}"
    questions_query_job = client.query(tag_query, job_config=safe_config)
    questions_results = questions_query_job.to_dataframe()
    return questions_results


@st.cache
def get_answer_tag_by_user(user_id):
    tag_query = f"SELECT q.tags \
    FROM `bigquery-public-data.stackoverflow.posts_questions` AS q \
    INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a \
        ON q.id = a.parent_id \
    WHERE a.owner_user_id IN {serialization(user_id)}"
    questions_query_job = client.query(tag_query, job_config=safe_config)
    questions_results = questions_query_job.to_dataframe()
    return questions_results


@st.cache
def process_tags(tag_list):
    tags = []
    for i in tag_list:
        tags += i.split('|')
    return Counter(tags)


@st.cache
def get_tag_freq_by_user(user_id):
    qt = get_question_tag_by_user(user_id)
    at = get_answer_tag_by_user(user_id)
    freq = process_tags(qt['tags'].to_list() + at['tags'].to_list())
    return freq


@st.cache(allow_output_mutation=True)
def get_user_interaction():
    # query = f"SELECT q.owner_user_id, a.owner_user_id, COUNT(1)\
    # FROM `bigquery-public-data.stackoverflow.posts_questions` AS q \
    # INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a \
    #     ON q.id = a.parent_id \
    # WHERE a.owner_user_id IN {serialization(user_id)} AND q.owner_user_id IN {serialization(user_id)} \
    # GROUP BY "
    users = pd.read_csv('users.csv')
    answers = pd.read_csv('answers.csv')
    questions = pd.read_csv('questions.csv')
    uu = np.zeros([len(users), len(users)])
    qid2uid = {}
    for qid, uid in zip(questions['id'], questions['owner_user_id']):
        qid2uid[qid] = uid
    for qid, uid in zip(answers['qid'], answers['uid']):
        try:
            uid2 = qid2uid[qid]
            uu[uid, uid2] += 1
            uu[uid2, uid] += 1
        except:
            continue
    return uu.max(axis=1)


@st.cache
def get_query(query):
    query_job = client.query(query, job_config = safe_config)
    query_results = query_job.to_dataframe()
    return query_results




df = get_all_users(reputation_thres=100000)
bigV = df['id'].to_list()
# st.write(len(bigV))

st.header("What's the most popular topics on Stack Overflow?")
freq = get_tag_freq_by_user(bigV)
wc = WordCloud(background_color="white", width=MAX_WIDTH * 2, height=400)
st.image(wc.generate_from_frequencies(freq).to_image(),
         use_column_width=True)

# u-u interaction in QA
p = get_user_interaction()
p.sort()
p = p[p > 0]
# fig, ax = plt.subplots()
# ax.plot(range(len(p)), p[::-1])
# st.pyplot(fig)




sample_users = get_query(f"SELECT id \
                            FROM `bigquery-public-data.stackoverflow.users` \
                            WHERE RAND() < 10/100000")["id"].to_list()

sample_users_repu = get_query(f"SELECT id, reputation, up_votes FROM `bigquery-public-data.stackoverflow.users` \
                            WHERE id IN {serialization(sample_users)} ORDER BY reputation DESC")


sample_users_repu_plot = alt.Chart(sample_users_repu).mark_bar().encode(
    y = 'reputation:Q',
    x = alt.X('id'),
)

# sample_users_repu_plot

answered_questions = get_answered_questions_by_user(sample_users)


tag_cnt = process_tags(answered_questions['tags'])

tag_cnt = sorted(tag_cnt.items(), key=lambda pair: pair[1], reverse=True)


tags = list()
count = list()
for tag, cnt in tag_cnt:
    tags.append(tag)
    count.append(cnt)

tag_cnt_df = pd.DataFrame({'tags': tags, 'count': count})

'The 15 most popular question tags from 0.01% samples:'
tag_cnt_plot = alt.Chart(tag_cnt_df[:15]).mark_bar().encode(
    x = alt.Y('tags:N', sort = '-y'),
    y = 'count',
).properties(
    width = 500,
    height = 300
)

tag_cnt_plot



tech = st.selectbox('How does the popularity of a specific technology change over years?',
                    ('neo4j', 'java', 'php', 'javascript', 'c#', 'c++', 'html', 'python', 'ruby', 'jquery', 'css', 'mysql', 'r', 'amazon-web-services', 'android'))

tech_popularity_per_year = get_query(f"SELECT EXTRACT(year FROM creation_date) AS year, COUNT(*) AS question_count \
        FROM `bigquery-public-data.stackoverflow.posts_questions` \
        WHERE tags LIKE '%{tech}%' \
        GROUP BY year \
        ORDER BY year")

fig, ax = plt.subplots(figsize=(14, 4))
sns.lineplot(x = 'year', y = 'question_count', data = tech_popularity_per_year)
st.pyplot(fig)


st.header("How active are users on Stack Overflow?")

'Answer rate of questions is decresing over the years.'

answer_rate_per_year = get_query("""
            SELECT EXTRACT(YEAR FROM creation_date) AS year,
                  ROUND(100 * SUM(IF(answer_count > 0, 1, 0)) / COUNT(*), 1) AS percentage_of_questions_with_answers
            FROM `bigquery-public-data.stackoverflow.posts_questions`
            GROUP BY year
            ORDER BY year
            """)

# fig, ax = plt.subplots(figsize=(14, 4))
# sns.barplot(x = 'year', y = 'percentage_of_questions_with_answers', data = answer_rate_per_year)
# st.pyplot(fig)



question_cnt_per_year = get_query("""
            SELECT EXTRACT(YEAR FROM creation_date) AS year, COUNT(*) AS number_of_questions
            FROM `bigquery-public-data.stackoverflow.posts_questions`
            GROUP BY year
            ORDER BY year
            """)

answered_question_cnt_per_year = get_query("""
            SELECT EXTRACT(YEAR FROM creation_date) AS year, COUNT(*) AS number_of_questions_with_answers
            FROM `bigquery-public-data.stackoverflow.posts_questions`
            WHERE answer_count > 0
            GROUP BY year
            ORDER BY year
            """)

user_cnt_per_year = get_query("""
            SELECT EXTRACT(YEAR FROM creation_date) AS year, COUNT(*) AS number_of_users
            FROM `bigquery-public-data.stackoverflow.users`
            GROUP BY year
            ORDER BY year
            """)
# user_cnt_per_year


fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (20, 10), sharex = False)

sns.set_color_codes("pastel")
sns.barplot(x = "year", y = "number_of_questions", data = question_cnt_per_year,
            label = "Total Questions", color = "b", ax = ax1)
sns.set_color_codes("muted")
sns.barplot(x = "year", y = "number_of_questions_with_answers", data = answered_question_cnt_per_year,
            label = "Questions have answers", color = "b", ax = ax1)
sns.set_color_codes("dark")
sns.lineplot(x = "year", y = "number_of_users", data = user_cnt_per_year,
            label = "Number of users", color = "b", ax = ax2)
# ax.legend(ncol=2, frameon=True)
# ax.set(ylabel="Question Count",
#        xlabel="Year")
st.pyplot(fig)



user_cnt_per_year = get_query("""
            SELECT EXTRACT(YEAR FROM creation_date) AS year, COUNT(*) AS number_of_users
            FROM `bigquery-public-data.stackoverflow.users`
            GROUP BY year
            ORDER BY year
            """)
# user_cnt_per_year



avg_repu_per_year = get_query("""
            SELECT  EXTRACT(YEAR FROM creation_date) AS date,
                    AVG(reputation) AS reputation
            FROM `bigquery-public-data.stackoverflow.users`
            GROUP BY date
            ORDER BY date
            """)

'Average user reputation is also decreasing.'
fig, ax = plt.subplots(figsize=(14, 4))
sns.barplot(x = 'date', y = 'reputation', data = avg_repu_per_year)
st.pyplot(fig)





questioner = get_query('''
                        select count(distinct q.owner_user_id)
                        from `bigquery-public-data.stackoverflow.posts_questions` q
                        left join `bigquery-public-data.stackoverflow.posts_answers` a
                        on q.owner_user_id = a.owner_user_id
                        where a.owner_user_id is null
                        ''').iat[0,0]

answerer = get_query('''
                        select count(distinct a.owner_user_id)
                        from `bigquery-public-data.stackoverflow.posts_answers` a
                        left join `bigquery-public-data.stackoverflow.posts_questions` q
                        on a.owner_user_id = q.owner_user_id
                        where q.owner_user_id is null
                        ''').iat[0,0]

question_and_answerer = get_query('''
                        select count( distinct q.owner_user_id)
                        from `bigquery-public-data.stackoverflow.posts_questions` q
                        inner join `bigquery-public-data.stackoverflow.posts_answers` a 
                        on q.owner_user_id = a.owner_user_id
                        ''').iat[0,0]

do_nothinger = get_query('''
                        select count(id)
                        from `bigquery-public-data.stackoverflow.users` u
                        left join (
                            select distinct owner_user_id
                            from `bigquery-public-data.stackoverflow.posts_answers`
                            union all
                            select distinct owner_user_id
                            from `bigquery-public-data.stackoverflow.posts_questions`) b
                        on u.id = b.owner_user_id
                        where b.owner_user_id is null
                        ''').iat[0,0]

num_user = get_query("select count(*) from `bigquery-public-data.stackoverflow.users` ").iat[0,0]

# Show result
user_type_df = pd.DataFrame({"Number of Users": [questioner, answerer, question_and_answerer, do_nothinger, num_user]})
user_type_df["Percentage(%)"] = round(user_type_df["Number of Users"] / num_user * 100, 2)
user_type_df.index = ["Questioner", "Answerer", "Question-and-answerer", "Do-nothinger", "Total"]
user_type_df.reset_index(inplace=True)
user_type_df.rename(columns = {'index': 'User Type'}, inplace=True)

'Over half of the users have never answered a question or posted a question!'
fig, ax = plt.subplots(figsize=(14, 4))
plot = sns.barplot(x = 'Number of Users', y = 'User Type', data = user_type_df)
st.pyplot(fig)


st.header('Objective')
st.write("increase user activity and question answer rates by recommending questions and users for users.")





# U-U

similar_users = pd.read_csv('similar_users.csv')[['id', 'display_name']]

st.header('Method:')
'Associate users with the tags of questions and answers they posted. Construct user-tag matrix to calculate correlation and similarity among users'

'For example, we can find the top 5 users that are similar to user with id 3430'

similar_users

st.header('TODO:')

'1. Recommend users to users based on similarity.'

"2. Recommend questions for users to answer based on the user's history of question and answer posts."

'3. Interactive visualization that demonstrates the recommendation process.'