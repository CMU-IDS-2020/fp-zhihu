import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz, hstack
from sklearn.metrics.pairwise import cosine_similarity
import os


class UserPool:
    def __init__(self, followings=None, verbose=False):
        user_path = './data/users.csv'
        tag_path = './data/tags.csv'
        question_path = './data/questions.csv'
        answer_path = './data/answers.csv'
        user_normalized_qtag_path = './data/user_normalized_qtag.npz'
        user_weighted_atag_path = './data/user_weighted_atag.npz'
        user_qtag_path = "./data/user_qtag.npz"
        user_atag_path = "./data/user_atag.npz"
        self.verbose = verbose
        self.epsilon = 1e-5
        user_df = pd.read_csv(user_path).dropna(subset=["id"])
        if self.verbose:
            print(f"{user_path} loaded")
        tag_df = pd.read_csv(tag_path).dropna(subset=["id", "tag_name"])
        if self.verbose:
            print(f"{tag_path} loaded")
        question_df = pd.read_csv(question_path)
        if self.verbose:
            print(f"{question_path} loaded")
        answer_df = pd.read_csv(answer_path)
        if self.verbose:
            print(f"{answer_path} loaded")
        user_id = user_df['id'].tolist()
        self.users = dict(zip(user_id, range(len(user_id))))
        self.index2id = dict(zip(range(len(user_id)), user_id))
        self.reputations = np.array(user_df['reputation'].fillna(0).tolist())
        self.names = user_df['display_name'].fillna("default_name").tolist()
        assert len(self.reputations) == len(self.names) == len(self.users)
        tag_id = tag_df['id'].tolist()
        tag_names = tag_df['tag_name'].tolist()
        assert len(tag_id) == len(tag_names)
        self.tags = dict(zip(tag_names, range(len(tag_id))))
        self.tag_id2index = dict(zip(tag_id, range(len(tag_id))))
        self.tag_index2id = tag_id

        if os.path.isfile(user_qtag_path):
            self.user_qtag = load_npz(user_qtag_path).toarray()
        else:
            self.user_qtag = self._get_user_qtag_matrix(question_df)
        if os.path.isfile(user_atag_path):
            self.user_atag = load_npz(user_atag_path).toarray()
        else:
            self.user_atag = self._get_user_atag_matrix(answer_df)
        self.followings = None
        self._init_followings(followings)
        self.user_weighted_atag = None
        self.user_normalized_qtag = None
        if os.path.isfile(user_normalized_qtag_path):
            self.user_normalized_qtag = load_npz(user_normalized_qtag_path).toarray()
        else:
            self.refresh_user_normalized_qtag()
            save_npz(user_normalized_qtag_path, csr_matrix(self.user_normalized_qtag))
        if os.path.isfile(user_weighted_atag_path):
            self.user_weighted_atag = load_npz(user_weighted_atag_path).toarray()
        else:
            self.refresh_user_weighted_atag()
            save_npz(user_weighted_atag_path, csr_matrix(self.user_weighted_atag))


    def get_reputation_by_id(self, id):
        return self.reputations[self.users[id]]

    def get_name_by_id(self, id):
        return self.names[self.users[id]]

    def get_followings_by_id(self, id, return_id=True):
        if return_id:
            return [self.index2id[index] for index in np.nonzero(self.followings[self.users[id]])[0]]
        else:
            return self.followings[self.users[id]]

    @property
    def user_set(self):
        return set(self.users.keys())

    @property
    def user_number(self):
        return len(self.users)

    def get_id_by_index(self, index):
        return self.index2id[index]

    def add_user(self, user_id, name="default_name"):
        self.users[user_id] = len(self.users)
        self.index2id[len(self.users) - 1] = user_id
        self.reputations = np.insert(self.reputations, -1, 0)
        self.names.append(name)
        self._init_followings()
        self.user_atag = np.vstack((self.user_atag, np.zeros(len(self.tags), dtype=np.int)))
        self.user_qtag = np.vstack([self.user_qtag, np.zeros(len(self.tags), dtype=np.int)])
        self.user_weighted_atag = np.vstack([self.user_weighted_atag, np.zeros(len(self.tags), dtype=np.float)])
        self.user_normalized_qtag = np.vstack([self.user_normalized_qtag, np.zeros(len(self.tags), dtype=np.float)])

    def add_followings(self, follower, following_list):
        if not isinstance(following_list, list):
            following_list = [following_list]
        for followee in following_list:
            self._add_following(follower, followee)

    def refresh_user_weighted_atag(self, id=None):
        if id is None:
            self.user_weighted_atag = self.user_atag.astype(np.float) / (
                        np.linalg.norm(self.user_atag, axis=1) + self.epsilon)[:, None] * self.reputations[:, None]
        else:
            index = self.users[id]
            self.user_weighted_atag[index] = self.user_atag[index].astype(np.float) / (
                    np.sum(self.user_atag[index] ** 2) + self.epsilon) * self.reputations[index]

    def refresh_user_normalized_qtag(self, id=None):
        if id is None:
            self.user_normalized_qtag = self.user_qtag.astype(np.float) / (
                    np.linalg.norm(self.user_qtag, axis=1) + self.epsilon)[:, None]
        else:
            index = self.users[id]
            self.user_normalized_qtag = self.user_qtag.astype(np.float) / (
                    np.sum(self.user_qtag[index] ** 2) + self.epsilon)

    def get_recommendation_score_by_id(self, id):
        scores = (self.user_weighted_atag * self.user_normalized_qtag[self.users[id]]).sum(axis=1)
        return scores

    def get_recommendation_by_tag_id(self, tag_id, k):
        if not isinstance(tag_id, list):
            tag_id = [tag_id]
        tag_index = [self.tag_id2index[id] for id in tag_id]
        mask = np.zeros(len(self.tags), dtype=np.bool)
        mask[tag_index] = True
        scores = (self.user_weighted_atag * mask).sum(axis=1)
        return [self.index2id[index] for index in np.argsort(scores)[-k:][::-1]]

    def _get_user_qtag_matrix(self, question_df):
        user_num = len(self.users)
        tags_num = len(self.tags)
        assert tags_num == 59456
        shape = (user_num, tags_num)
        matrix = np.zeros(shape, dtype=np.int)

        for user_index, user_id in tqdm(enumerate(self.users), disable=~self.verbose):
            questions_tags = question_df[question_df['owner_user_id'] == user_id]['tags'].to_list()
            for q_tag in questions_tags:
                q_tag = str(q_tag)
                for tag_name in q_tag.split('|'):
                    if tag_name != "null" and tag_name != "nan":
                        matrix[user_index][self.tags[tag_name]] += 1

        return matrix

    def _get_user_atag_matrix(self, answer_df):
        user_num = len(self.users)
        tags_num = len(self.tags)
        assert tags_num == 59456
        shape = (user_num, tags_num)
        matrix = np.zeros(shape, dtype=np.int)

        for user_index, user_id in tqdm(enumerate(self.users), disable=~self.verbose):
            answered_questions = answer_df[answer_df['uid'] == user_id]['qtags'].to_list()

            for a_tag in answered_questions:
                a_tag = str(a_tag)
                for tag_name in a_tag.split('|'):
                    if tag_name != "null" and tag_name != "nan":
                        matrix[user_index][self.tags[tag_name]] += 1

        return matrix

    def _init_followings(self, followings=None):
        if self.followings is None:
            if followings is None:
                self.followings = np.eye(len(self.users), dtype=np.bool)
            else:
                self.followings = followings
        else:
            self.followings = np.insert(self.followings, -1, values=False, axis=1)
            self.followings = np.insert(self.followings, -1, values=False, axis=0)
            self.followings[-1][-1] = True

    def _add_following(self, follower, followee):
        self.followings[self.users[follower]][self.users[followee]] = True


class User:
    def __init__(self, user_pool, id, name=None):
        self.user_pool = user_pool
        self.id = id
        if id not in self.user_pool.user_set:
            self.user_pool.add_user(user_id=id, name=name)

    def recommend_users_by_history(self, k):
        scores = self.user_pool.get_recommendation_score_by_id(self.id)
        while True:
            followings = self.user_pool.get_followings_by_id(id=self.id, return_id=False)
            stranger_index, = np.where(~followings)
            candidate_index = stranger_index[np.argsort(scores[stranger_index])[-k:][::-1]]
            yield [self.user_pool.get_id_by_index(index) for index in candidate_index]

    def add_followings(self, followee):
        self.user_pool.add_followings(self.id, followee)

    @property
    def followings(self):
        return self.user_pool.get_followings_by_id(self.id)

    @property
    def reputation(self):
        return self.user_pool.get_reputation_by_id(self.id)


if __name__ == "__main__":
    user_pool = UserPool(verbose=True)
    # (21209, 59456)
    # user_id[:10] = [3460, 3578, 5462, 11196, 24875, 54486, 61675, 67865, 98016, 143078]
    # tag_id[:5] = [1327, 13399, 19023, 20397, 23386】
    print(f"total number of users {user_pool.user_number}")
    user = User(user_pool=user_pool, id=3460)
    print(f"reputation {user.reputation}")
    print(f"followings {user.followings}")
    user_1 = User(user_pool, id=-1)
    print("user 1 added")
    user.add_followings([-1])
    print("user 1 followed")
    print(f"total number of users{user_pool.user_number}")
    print(f"followings {user.followings}")
    candidate_generator = user.recommend_users_by_history(5)
    i = 0
    for candidates in candidate_generator:
        if i > 3:
            break
        print(candidates)
        user.add_followings(candidates[0])
        i += 1
    print(user_pool.get_recommendation_by_tag_id([1327, 13399, 19023], k=10))
