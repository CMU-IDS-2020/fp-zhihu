from google.cloud import bigquery


# global variables definition

MAX_WIDTH = 700

PREFIX = 'bigquery-public-data.stackoverflow'

# user info metadata
table_name = 'badges comments posts_answers posts_questions users'.split()
user_id_str = 'user_id user_id owner_user_id owner_user_id id'.split()
date_str = 'date creation_date creation_date creation_date /'.split()

# # Create a "Client" object
# client = bigquery.Client()

# # Construct a reference to the "stackoverflow" dataset
# dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")

# # API request - fetch the dataset
# dataset = client.get_dataset(dataset_ref)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**11)
