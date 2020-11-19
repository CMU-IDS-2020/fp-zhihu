import altair as alt
import numpy as np
import os
import pandas as pd
import streamlit as st

from google.cloud import bigquery

# global variables definition

# update the secret to env
# this credential file should not commit to github
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcloud.json'

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "stackoverflow" dataset
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**11)


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
def get_all_users(location=None, user_id=None):
    if location:
        questions_query = f"SELECT id, display_name, reputation, up_votes, down_votes FROM `bigquery-public-data.stackoverflow.users` WHERE location IN {serialization(location)}"
    elif user_id:
        questions_query = f"SELECT id, display_name, reputation, up_votes, down_votes FROM `bigquery-public-data.stackoverflow.users` WHERE id IN {serialization(user_id)}"
    else:
        questions_query = f"SELECT id, display_name, reputation, up_votes, down_votes FROM `bigquery-public-data.stackoverflow.users`"

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
    questions_query = f"SELECT a.id, a.body, a.owner_user_id AS uid, q.id AS qid \
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


st.write(get_table('users', 3))
st.write(get_answered_questions_by_user(3043))
st.write(get_all_users(user_id=[3043, 2493]))
