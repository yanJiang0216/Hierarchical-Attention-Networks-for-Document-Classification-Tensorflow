import os
from os import walk
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec

def gen_formatted_review(data_dir, tokenizer = RegexpTokenizer(r'\w+') ):
    data = []
    for  filename in os.listdir(data_dir):
        file = os.path.join(data_dir, filename)
        with open(file,encoding='utf-8') as f:
            content = f.readline().lower()
            content_formatted = tokenizer.tokenize(content)
            data.append(content_formatted)
    return data
# def gen_Yelp_formatted_review(data_dir):


if __name__ == "__main__":

    # create IMDB word vector
    embedding_size =200
    working_dir = "./data/aclImdb"
    train_dir = os.path.join(working_dir, "train")
    train_pos_dir = os.path.join(train_dir, "pos")
    train_neg_dir = os.path.join(train_dir, "neg")
    test_dir = os.path.join(working_dir, "test")
    test_pos_dir = os.path.join(test_dir, "pos")
    test_neg_dir = os.path.join(test_dir, "neg")
    train = gen_formatted_review(train_pos_dir)
    train2 = gen_formatted_review(train_neg_dir)
    train.extend(train2)
    test = gen_formatted_review(test_pos_dir)
    test2 = gen_formatted_review(test_neg_dir)
    test.extend(test2)
    train.extend(test)
    corpus_names=[train,train2]
    emmbedding_model = os.path.join(working_dir, "imdb_embedding")
    for corpus_name in corpus_names:
        if os.path.isfile(emmbedding_model):
            embedding_model = Word2Vec.load(emmbedding_model)
            embedding_model = Word2Vec(corpus_name, size=embedding_size, window=5, min_count=5)
            embedding_model.save(emmbedding_model)
        else:
            embedding_model = Word2Vec(corpus_name, size=embedding_size, window=5, min_count=5)
            embedding_model.save(emmbedding_model)
    #create Yelp_2015 word_embedding
    # word1 = "great"
    # word2 = "horrible"
    # print("similar words of {}:".format(word1))
    # print(embedding_model.most_similar('great'))
    # print("similar words of {}:".format(word2))
    # print(embedding_model.most_similar('horrible'))
    pass
