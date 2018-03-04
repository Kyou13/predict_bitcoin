# -*- coding:utf-8 -*-

# import 
import os 
import sys
import MeCab
import collections
import argparse
import re
import datetime as dt
import numpy as np
from gensim import models
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess as preprocess

# directory以下の全ファイル取得
def get_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            # generator
            yield os.path.join(root, file)

# ファイルから文章取得
# 文章整形
def read_docment(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    return lines 

def trim_doc(lines,name):
    sep_docs = collections.OrderedDict()
    sep_doc = []

    with open("./tweet_date_list/testfile_{}.txt".format(name),'w') as f:

        # 文章ごとに区切る
        for s in range(len(lines)):
            # 空白行だったら
            if lines[s] == '\n':
                sep_docs[tag] = ''.join(sep_doc)
                sep_doc = []

            # 日時情報取得
            elif lines[s][-5:-2] == "201":
                day = lines[s][4:19]
                tag = day + " @ " + name
                f.write("{}\n".format(tag))
                continue

            # ツイート文章
            else :
                # URLを除去
                lines[s] = re.sub('https?://[\w/:%#\$&\?\(\)~\.=\+\-]+','',lines[s])

            sep_doc.append(lines[s])
    
    return sep_docs

    
# 単語に分解
# TaggedDocumentのTagは日付
def split_into_words(doc, name=''):
    mecab = MeCab.Tagger("-Ochasen")
    lines = mecab.parse(doc).splitlines()
    words = []
    for line in lines:
        chunks = line.split('\t')
        # 動詞,形容詞,名詞のみを抽出
        if len(chunks) > 3 and (chunks[3].startswith('動詞') or chunks[3].startswith('形容詞') or (chunks[3].startswith('名詞') and not chunks[3].startswith('名詞-数'))):
            words.append(chunks[0])
    print(words)
    return TaggedDocument(words=words, tags=[name])

def corpus_to_sentences(docs, names):
    for idx,(doc, name) in enumerate(zip(docs, names)):
        sys.stdout.write('\rProcessing:{} {}'.format(idx, doc))
        yield split_into_words(doc, name)

# モデルトレーニング
def train(sentences):
    ###
    # size:ベクトル化したときの次元数
    # alpha:学習率
    # min_couns: 学習に使う単語の最低出現回数
    # workers: 並列実行数
    ###

    # doc2vec学習条件
    model = models.Doc2Vec(size=400, alpha=0.0015, min_count=1,workers=4, iter=50)
    # doc2vecの学習前準備(単語リスト構築)
    model.build_vocab(sentences)
    # 学習
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    print("\nTraining Finish")
    # print(model.infer_vector(preprocess("This is a document.")))

    return model

# 類似文章推定
def search_similar_texts(words):

    model = models.Doc2Vec.load('doc2vec.model')
    model.seed = 1
    x = model.infer_vector(words)
    # 引数はid
    most_similar_texts = model.docvecs.most_similar([x])
    for similar_text in most_similar_texts:
        print(similar_text[0], similar_text[1])
        # ベクトル表示
        # print(model.infer_vector(similar_text[0]))

def similarity_texts(words):

    model = models.Doc2Vec.load('doc2vec.model')
    with open('testfile.txt') as f:
        tag_names=f.readlines()

    # tweet_index
    tweet_index = find_tweet_date(tag_names)

    tweet_vec = np.empty((0,400),np.float32)

    for index in tweet_index:
        if index != 0:
            for tag_name in tag_names:
                tag_name_ = tag_name.split("@")
                tag_date = dt.datetime.strptime(tag_name_[0],"%b %d %H:%M:%S ")
                tag_date = tag_date.replace(year=2017)
                if index.day == tag_date.day and index.month == tag_date.month:
                    tweet_vec = np.append(tweet_vec,model.infer_vector(tag_name).reshape((1,400)),axis=0)
                    break

        else:
            tweet_vec = np.append(tweet_vec,np.zeros((1,400)),axis=0)

    # numpy配列保存
    np.save("tweet_vec.npy",tweet_vec)


# ツイートした日していない日を区別するindexを作成
def find_tweet_date(tag_names):

    # 2017年の日付を全て挿入したリスト
    year_list = date_year()
    tweet_list = []

    for date in year_list:

        day_find = False
        for tag_name in tag_names:
            tag_name = tag_name.split("@")
            tag_date = dt.datetime.strptime(tag_name[0],"%b %d %H:%M:%S ")
            if tag_date.month >= 8:
                tag_date = tag_date.replace(year=2017)
            else :
                tag_date = tag_date.replace(year=2018)
            if date.day == tag_date.day and date.month == tag_date.month:
                tweet_list.append(tag_date)
                day_find = True
                break
        if day_find == False:
            tweet_list.append(0)

    return tweet_list

def date_year():
    date = dt.datetime(2018, 2, 18, 0, 0)
    year_list = []
    for i in range(180):
        year_list.append(date)
        date -= dt.timedelta(days=1)

    year_list = year_list[::-1]
    return year_list

# 類似単語推定
def search_similar_words(word):

    #for word in words:
    if True:
        model = models.Doc2Vec.load('doc2vec.model')
        print()
        print(word + ':')
        for result in model.most_similar(word, topn=10):
            print(result[0],result[1])

def add_text():

    with open('./testfile.txt', 'w') as outfile:
        for i in get_all_files('./tweet_date_list'):
            with open(i) as infile:
                outfile.write(infile.read())




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('--t',action='store_false') 

    args = parser.parse_args()

    if args.t is False : 
        for i in get_all_files('./database'):
            name = i.split('/')[2][:-4]
            raw_docs = read_docment(i) # list
            doc = trim_doc(raw_docs,name)

            # ローカルスコープでチェック
            # 更新する
            if not 'docs' in locals():
                docs = doc
            else:
                docs.update(doc)
        # .txt連結
        add_text()

        # ジェネレータ
        sentences = corpus_to_sentences(docs.values(),docs.keys())

        model = train(sentences)
        model.save('doc2vec_multiple_user_test.model')
    else:
        test_word = "ビットコイン買う"
        
        search_similar_texts(test_word)
        # npyとして保存
        similarity_texts(test_word)
        # search_similar_words(test_word)
