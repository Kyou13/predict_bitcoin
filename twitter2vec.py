# -*- coding:utf-8 -*-

# import 
import os 
import sys
import MeCab
import collections
import argparse
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
def read_docment(path, name):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    sep_docs = collections.OrderedDict()
    sep_doc = []

    for s in range(len(lines)):
        if lines[s] == '\n':
            sep_docs[tag] = ''.join(sep_doc)
            sep_doc = []

        # 日時情報取得
        if lines[s][-5:-2] == "201":
            day = lines[s][4:19]
            tag = day + " @ " + name
            continue

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
    return TaggedDocument(words=words, tags=[name])

def corpus_to_sentences(docs, names):
    for idx,(doc, name) in enumerate(zip(docs, names)):
        sys.stdout.write('\r前処理中{} / {}'.format(idx, doc))
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
    model = models.Doc2Vec(size=300, alpha=0.0015, min_count=1,workers=4)
    # doc2vecの学習前準備(単語リスト構築)
    model.build_vocab(sentences)
    # 学習
    model.train(sentences, total_examples=model.corpus_count, epochs=30)

    print("Training Finish")
    # print(model.infer_vector(preprocess("This is a document.")))
        

    return model

def search_similar_texts(words):

    model = models.Doc2Vec.load('doc2vec.model')
    x = model.infer_vector(words)
    most_similar_texts = model.docvecs.most_similar([x])
    for similar_text in most_similar_texts:
        print(similar_text[0])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('--t',action='store_true') 

    args = parser.parse_args()

    if args.t is False : 
        for i in get_all_files('./database'):
            name = i.split('/')[2][:-4]
            docs = read_docment(i,name)

        # ジェネレータ
        sentences = corpus_to_sentences(docs.values(),docs.keys())

        model = train(sentences)
        model.save('doc2vec.model')
    else:
        search_similar_texts("ビットコイン")
