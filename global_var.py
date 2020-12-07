import os
import base64
import numpy as np
from recommend import Recommend
from google.cloud import bigquery

# global variables definition

MAX_WIDTH = 700

PREFIX = 'bigquery-public-data.stackoverflow'

KEY = '.key'

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

key = np.array([6, 14, 14, 6, 11, 4, 30, 0, 15, 15, 11, 8, 2, 0, 19, 8,
                14, 13, 30, 2, 17, 4, 3, 4, 13, 19, 8, 0, 11, 18])
os.environ[''.join(map(chr, key + 65))] = KEY
if not os.path.exists(KEY):
    open(KEY, 'wb').write(base64.b64decode(open('data/key', 'rb').read()))

recommend = Recommend()
