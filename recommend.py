import numpy as np
import pandas as pd
from scipy.sparse import load_npz


class Recommend:
    def __init__(self):
        tag_df = pd.read_csv('data/tags.csv').dropna(subset=["id", "tag_name"])
        tag_id = tag_df['id'].to_numpy()
        tag_names = tag_df['tag_name'].to_list()
        self.tag_id2index = np.zeros(tag_id.max() + 1, np.int)
        for i, t in enumerate(tag_id):
            self.tag_id2index[t] = i
        self.tag_index2id = tag_id
        self.tag_name2index = {}
        self.tag_index2name = np.array(["" for i in range(tag_id.max() + 1)], np.object)
        for i, n in enumerate(tag_names):
            self.tag_name2index[n] = i
            self.tag_index2name[i] = n

        self.user_weighted_atag = load_npz(
            'data/user_weighted_atag.npz').toarray()
        self.user_normalized_qtag = load_npz(
            'data/user_normalized_qtag.npz').toarray()

        self.index2id = np.load('data/user_id.npy')
        self.users = np.zeros(self.index2id.max() + 1, np.int)
        for i, t in enumerate(self.index2id):
            self.users[t] = i

        self.followings = {}

    def get_recommendation_by_tag_id(self, tag_id, k):
        tag_index = self.tag_id2index[tag_id]
        scores = self.user_weighted_atag[:, tag_index].sum(axis=1)
        return self.index2id[np.argsort(scores)[-k:][::-1]]

    def get_recommendation_score_by_id(self, id):
        return self.user_weighted_atag.dot(
            self.user_normalized_qtag[self.users[id]])

    def get_followings_by_id(self, id, return_id=True):
        index = np.array(self.followings.get(
            self.users[id], []) + [self.users[id]])
        if return_id:
            index = self.index2id[index]
        return index

    def get_id_by_index(self, index):
        return self.index2id[index]

    def recommend_users_by_history(self, id, k):
        scores = self.get_recommendation_score_by_id(id)
        followings = self.get_followings_by_id(id=id, return_id=False)
        scores[followings] = -np.inf
        candidate_index = np.argsort(scores)[-k:][::-1]
        return self.index2id[candidate_index]

    def add_followings(self, follower, following_list):
        if not isinstance(following_list, list):
            following_list = [following_list]
        key = self.users[follower]
        followee = self.users[following_list].tolist()
        self.followings[key] = self.followings.get(key, []) + followee

    def get_tag_id_by_name(self, name_list) -> np.ndarray:
        if not isinstance(name_list, list):
            name_list = [name_list]
        result = []
        for n in name_list:
            if n in self.tag_name2index:
                result.append(self.tag_index2id[self.tag_name2index[n]])
        return np.array(result)

    def get_tag_name_by_id(self, id) -> list:
        index = self.tag_id2index[id]
        result = self.tag_index2name[index].tolist()
        return [i for i in result if i]


if __name__ == '__main__':
    r = Recommend()
    print(r.get_recommendation_by_tag_id([1327, 13399, 19023], k=10))
    print(r.get_recommendation_score_by_id(3460))
    for i in range(5):
        candidates = r.recommend_users_by_history(3460, 5)
        print(candidates)
        r.add_followings(3460, candidates[0])
    name_str = r.get_tag_name_by_id([1327, 13399, 19023])
    print(name_str)  # [0, 1, 2]
    print(r.get_tag_id_by_name(name_str))  # [1327, 13399, 19023]
