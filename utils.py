import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nfuncs = 10 # 映射函数数量
userid = 671
K = 20


# 读取movies数据
movies = pd.read_csv('datasets/movies.csv')
# 生成一个0开始的连续下标和movieId的双向映射
index_Id = {k:v for k, v in enumerate(movies['movieId'])}
Id_index = {v:k for k, v in index_Id.items()}


def process():
    genres = [' '.join(movies['genres'][i].split('|')) for i in range(len(movies))]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(genres).toarray()
    matrix = pd.DataFrame(matrix)
    cosine_matrix = cosine_similarity(matrix)
    return cosine_matrix


def process_minihash():
    genres = [' '.join(movies['genres'][i].split('|')) for i in range(len(movies))]
    # 如果使用minhash生成01矩阵
    cnt = CountVectorizer(binary=True)
    matrix = cnt.fit_transform(genres).toarray().T  # 此处转置获得(features_num * movies_num)大小的矩阵，便于后续优化
    matrix = pd.DataFrame(matrix)
    matrix.columns = list(index_Id.values())

    # 根据随机生成的nfuncs个映射函数生成哈希签名矩阵
    features_num = len(matrix)
    movies_num = len(matrix.columns)

    sig_matrix = np.zeros((nfuncs, movies_num))#构造降维后的矩阵

    for i in range(nfuncs):#根据所要降的纬度，循环求出每一个维度上的值，具体方法就是：
        func = list(range(1, features_num + 1))
        random.shuffle(func)  # 先对原特征shuffle

        k = dict(zip(func, [np.array(matrix.loc[i]) for i in range(features_num)]))
        s = set(range(movies_num))  # 记录对于每个func，feature是否找到第一个1的集合，当feature找到了则从集合中弹出

        sig_i = np.zeros(movies_num)
        for j in range(1, features_num + 1):#对每一个原特征
            row = k[j]#这里就是对应 shuffle 后的特征
            for r in range(movies_num):#对每一部电影
                if row[r] and r in s:#不要重复赋值
                    s.remove(r)
                    sig_i[r] = j#取第一个非零特征位置数作为降维后的特征
            if not s:
                break

        sig_matrix[i] = sig_i  # 更新签名矩阵的第i行

    return pd.DataFrame(sig_matrix)



def cal_score(rated_movies, rating, method, movieid):
    cosine_matrix = process()
    sig_matrix = process_minihash()
    # cosine
    if method == 'cosine':

        distances = cosine_matrix[movieid]  # 从movieid出发的距离向量
        computed_dict = {}  # 计算集合
        for i in range(len(rated_movies)):#对已经评过分的电影中找与movieID相似的
            rated_movie = rated_movies[i]
            cosine = distances[Id_index[rated_movie]]
            if cosine > 1e-6:#收集与电影movieid相似度高的电影
                computed_dict[i] = cosine#某一部评过分的电影k，与movieID相似度cosine

        # 计算集合不为空
        if len(computed_dict.keys()):
            score = 0
            sum_v0 = 0
            for k, v in computed_dict.items():
                score += rating[k] * v#根据与movieID相似的用户评分过的电影，加权求和，这里的k与v，都是用户真实的
                sum_v0 += v

            return score / sum_v0#求相似度高的电影的评分加权和

        # 计算集合为空
        else:
            return np.mean(rating)

    # jaccard
    elif method == 'jaccard':

        computed_dict = {}  # 计算集合
        for i in range(len(rated_movies)):#对已经评过分的电影中找与movieID相似的
            rated_movie = rated_movies[i]
            sim = np.sum(sig_matrix[movieid] == sig_matrix[Id_index[rated_movie]]) / nfuncs
            if sim > 1e-6:
                computed_dict[i] = sim

        # 计算集合不为空
        if len(computed_dict.keys()):
            score = 0
            sum_v0 = 0
            for k, v in computed_dict.items():
                score += rating[k] * v
                sum_v0 += v

            return score / sum_v0

        # 计算集合为空
        else:
            return np.mean(rating)
    else:
        raise Exception("Only Cosine and Jaccard are accepted.")


def recommender(train, mode, minhash, *args):

    # 直接预测模式
    if mode == 1:
        userid, movieid = args

    # topK模式
    else:
        userid, K = args

    # 获得当前用户的数据
    data = train[train['userId'] == userid]
    rated_movies = np.array(data['movieId'])
    rating = np.array(data['rating'])

    # 使用minhash优化
    if minhash:
        # 直接预测模式
        if mode == 1:
            return cal_score(rated_movies, rating, "jaccard", Id_index[movieid])

        # topK模式
        else:
            scores_dict = {}
            movies_num = len(index_Id)
            for i in range(movies_num):#每次输入一个电影，计算评分
                if i % 200 == 0:
                    print('%d/%d...' % (i + 1, movies_num))
                scores_dict[i] = cal_score(rated_movies, rating, "jaccard", i)#计算所有电影预估评分，根据用户所评分过的电影

            scores_list = sorted(scores_dict.items(), key=lambda d: d[1], reverse=True)
            print('User %d, the top %d recommendations are shown below:' % (userid, K))
            print('-----------------------------------------------------------')
            for i in range(K):#这里推荐的就是与用户评分过的电影相似的电影
                ind, score = scores_list[i]
                print('%6d | %70s | %.4f' % (index_Id[ind], movies['title'][ind], score))

    # 不使用minhash优化
    else:
        # 直接预测模式
        if mode == 1:
            return cal_score(rated_movies, rating, "cosine", Id_index[movieid])#直接返回预估评分值

        # topK模式
        else:
            scores_dict = {}
            movies_num = len(index_Id)
            for i in range(movies_num):
                if i % 200 == 0:
                    print('%d/%d...' % (i+1, movies_num))
                scores_dict[i] = cal_score(rated_movies, rating, "cosine", i)#这里是计算每一个movie的预估评分，返回最高的n个

            scores_list = sorted(scores_dict.items(), key=lambda d: d[1], reverse=True)
            print('User %d, the top %d recommendations are shown below:' % (userid, K))
            print('-----------------------------------------------------------')
            for i in range(K):
                ind, score = scores_list[i]
                print('%6d | %70s | %.4f' % (index_Id[ind], movies['title'][ind], score))
