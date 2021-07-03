import sys
from string import punctuation
import re
from collections import defaultdict
from pyspark import SparkContext
import sys
from math import ceil, floor, log2
import math
from pyspark import SparkConf
from itertools import islice, combinations, chain
import time
import json
import random

class task2:
    
    def createBusinessProfile(self, sc, extracted_data_rdd, stopwords, model_file_path):
        punct_numbers_regex = re.compile("[" + re.escape(punctuation) + "\d]")
        # To Do: Tests!
        def cleanText(texts, stop_words):
            list_of_words = []
            for text in texts:
                text = text.lower()
                text = re.sub(punct_numbers_regex, "", text)
                word_list = re.split(r"[\n\s\r]+", text)
                
                for word in word_list:
                    if word is not None and word != "" and word not in stop_words:
                        list_of_words.append(word)
            return list_of_words
        
        user_index_rdd = extracted_data_rdd\
            .map(lambda record: record[0])\
            .distinct()\
            .zipWithIndex()
    
        bid_index_rdd = extracted_data_rdd\
            .map(lambda record: record[1])\
            .distinct()\
            .zipWithIndex()

        # Define Dictionaries
        user_index_dict = user_index_rdd.collectAsMap()
        bid_index_dict = bid_index_rdd.collectAsMap()
        index_to_bid_map = {v: k for k, v in bid_index_dict.items()}
        index_to_uid_map = {v: k for k, v in user_index_dict.items()}

        bid_wds_rdd = extracted_data_rdd\
            .map(lambda record: (bid_index_dict[record[1]], str(record[2])))\
            .groupByKey()\
            .mapValues(lambda reviews_list: cleanText(list(reviews_list), set(stopwords)))

        def generate_IDF(word, bids, N):
            res = []
            n_i = len(bids)
            for bid in bids:
                idf_value = math.log2(N / n_i)
                res.append((bid, word, idf_value))
            return res

        def get_tf(words):
            d = defaultdict(int)
            maximum_frequency = 0
            for word in words:
                d[word] += 1
                if d[word] > maximum_frequency:
                    maximum_frequency = d[word]
            res = []
            for word, cnt in d.items():
                res.append((word, cnt / maximum_frequency))
            return res

        # # (business_id, word, tf_value)
        bid_wds_tf = bid_wds_rdd\
            .mapValues(lambda val: get_tf(val)) \
            .flatMap(lambda record: [(record[0], lis) for lis in record[1]])\
            .map(lambda record: (record[0], record[1][0], record[1][1]))

        number_of_docs = len(bid_index_dict)
        # (word -> idf_value)
        idf_values = bid_wds_rdd\
            .flatMap(lambda tup: [(word, tup[0]) for word in tup[1]])\
            .groupByKey()\
            .mapValues(lambda tup: math.log2(number_of_docs / len(set(tup))))\
            .collectAsMap()
        def calculate_tfidf(business_id, word, tf_value, idf_values):
            idf_val = idf_values[word]
            return (word, tf_value * idf_val)
        def get_sorted_tf_idfs(bid_word_tfidf_list):
            arr = sorted(list(bid_word_tfidf_list), reverse=True, key=lambda tup: tup[1])[:200]
            return arr
        business_profile = bid_wds_tf\
            .map(lambda tup: (tup[0], calculate_tfidf(tup[0], tup[1], tup[2], idf_values)))\
            .groupByKey()\
            .map(lambda tup: (index_to_bid_map[tup[0]], get_sorted_tf_idfs(tup[1]))).collect()
        business_dict = defaultdict(list)
        for business_words_tuple in business_profile:
            business_dict[business_words_tuple[0]] = business_words_tuple[1]
        business_res = []
        for obj in business_profile:
            curr = {}
            curr["bid"] = obj[0]
            curr["words"] = [business_list[0] for business_list in obj[1]]
            curr['t'] = "b"
            business_res.append(curr)

        def get_words_set(business_list, business_dict):
            word_list = []
            for business in business_list:
                word_list.extend(business_dict[business])
            return word_list

        user_profile = extracted_data_rdd\
            .map(lambda raw_data_tuple: (user_index_dict[raw_data_tuple[0]], raw_data_tuple[1]))\
            .groupByKey()\
            .mapValues(lambda business_list: list(set(business_list)))\
            .mapValues(lambda business_list: get_words_set(business_list, business_dict))\
            .map(lambda kv: (index_to_uid_map[kv[0]], [lis[0] for lis in kv[1][:600]])).collect()
        
        user_data = []
        for obj in user_profile:
            curr = {}
            curr["uid"] = obj[0]
            curr["words"] = obj[1]
            curr["t"] = "u"
        with open(model_file_path, "w+") as of:
            for obj in business_res:
                of.write(json.dumps(obj) + "\n")
            for obj in user_data:
                of.write(json.dumps(obj) + "\n")
        of.close()

        

    def runTask(self, sc, extracted_data_rdd, stopwords, model_file_path):
        self.createBusinessProfile(sc, extracted_data_rdd, stopwords, model_file_path)

    def processInput(self, sc, train_file_path, model_file_path, stopwords_path):
        extracted_data_rdd = sc.textFile(train_file_path)\
            .map(lambda row: json.loads(row))\
            .map(lambda row: (row['user_id'], row['business_id'], row['text']))

        with open(stopwords_path, 'r') as f:
            stopwords = f.readlines()
            stopwords = set([word.rstrip() for word in stopwords])

        self.runTask(sc, extracted_data_rdd, stopwords, model_file_path)

    def main(self):
        train_file_path = sys.argv[1]
        model_file_path = sys.argv[2]
        stopwords_path = sys.argv[3]
        conf = SparkConf().setAppName("task1").setMaster("local[*]")
        sc = SparkContext(conf=conf).getOrCreate()
        sc.setLogLevel("OFF")
        self.processInput(sc, train_file_path, model_file_path, stopwords_path)

if __name__ == "__main__":
    t2 = task2()
    t2.main()