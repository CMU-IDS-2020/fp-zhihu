import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz, hstack
from sklearn.metrics.pairwise import cosine_similarity
import os


class UserUserSim:

    def __init__(self, reload=False):
        user_path = 'users.csv'
        tag_path = 'tags.csv'
        question_path = 'questions.csv'
        answer_path = 'answers.csv'
        user_qtag_path = "user_qtag.npz"
        user_atag_path = "user_atag.npz"
        assert [os.path.isfile(path) for path in [user_path, tag_path, question_path, answer_path]] == [True] * 4
        self.users = pd.read_csv(user_path)
        self.tags = pd.read_csv(tag_path)
        self.questions = pd.read_csv(question_path)
        self.answers = pd.read_csv(answer_path)
        if reload or not os.path.isfile(user_qtag_path) or not os.path.isfile(user_qtag_path):
            self._save_user_tag(self._get_user_qtag_matrix(), user_qtag_path)
            self._save_user_tag(self._get_user_atag_matrix(), user_atag_path)
        self.user_atag = load_npz(user_qtag_path)
        self.user_qtag = load_npz(user_atag_path)
        self.sim = None

    def calc_sim(self):
        features = hstack([self.user_qtag, self.user_atag])
        self.sim = cosine_similarity(features)
        return self.sim

    def topk_sim_peers(self, user_id, k=5, thres=-1):
        u_index = self.users.index[self.users['id'] == user_id].tolist()[0]
        if self.sim is None:
            self.calc_sim()
        arr = self.sim[u_index]
        idx, = np.where(arr >= thres)
        idx = idx[idx != u_index]
        similar_index = idx[np.argsort(arr[idx])[-k:][::-1]]
        similarity = arr[similar_index]
        similar_ids = self.users.loc[list(similar_index), ['id', 'display_name']]
        return similar_ids, similarity

    def _get_user_qtag_matrix(self):
        user_num = len(self.users)
        tags_num = len(self.tags)

        shape = (user_num, tags_num)
        rows, cols = user_num, tags_num
        data = []

        for i in tqdm(range(user_num), desc="Calculating user-tag"):
            user_id = self.users['id'].iloc[i]
            questions_tags = self.questions[self.questions['owner_user_id'] == user_id]['tags'].to_list()
            tags_dict = collections.Counter()
            for q in questions_tags:
                q = str(q)
                for ele in q.split('|'):
                    tags_dict[ele] += 1

            for t in tags_dict:
                data.append([user_id, t, tags_dict[t]])

        df = pd.DataFrame(data, columns=['uid', 'tag', 'value'])
        df = df.pivot_table(index="uid", columns="tag", values='value')
        df = df.fillna(0)
        return df

    def _get_user_atag_matrix(self):
        user_num = len(self.users)
        tags_num = len(self.tags)

        shape = (user_num, tags_num)
        rows, cols = user_num, tags_num
        data = []

        for i in tqdm(range(user_num)):
            user_id = self.users['id'].iloc[i]
            tags_dict = collections.Counter()
            answered_questions = self.answers[self.answers['uid'] == user_id]['qtags'].to_list()

            for a in answered_questions:
                a = str(a)
                for ele in a.split('|'):
                    tags_dict[ele] += 1

            for t in tags_dict:
                data.append([user_id, t, tags_dict[t]])

        df = pd.DataFrame(data, columns=['uid', 'tag', 'value'])
        df = df.pivot_table(index="uid", columns="tag", values='value')
        df = df.fillna(0)
        return df

    def _save_user_tag(self, df, output_name):
        users_id = self.users.loc[:, ['id']]
        df = df.drop(['uid'], axis=1)
        matrix = users_id.merge(df, left_on="id", right_on="uid", how="left")
        matrix = matrix.fillna(0)
        matrix = matrix.iloc[:, 1:].to_numpy()
        save_npz(output_name, csr_matrix(matrix))


if __name__ == "__main__":
    uus = UserUserSim()
    sim_peers, similarity = uus.topk_sim_peers(user_id=3460, k=5, thres=0.5)
    print(sim_peers)
    print(similarity)
