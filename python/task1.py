from collections import defaultdict
from pyspark import SparkContext
import sys
from math import ceil, floor
from pyspark import SparkConf
from itertools import islice, combinations, chain
import time
import json
import random

class task1:
    
    def make_function(self, a, b, m):
        def hash_func(x_input):
            return ((a * x_input + b) % 1000000009) % m
        return hash_func

    def processInput(self, sc, input_file_path, output_file_path):
        extracted_uid_bid_rdd = sc.textFile(input_file_path) \
            .map(lambda row: json.loads(row)) \
            .map(lambda record: (record['user_id'], record['business_id'])).cache()
        
        user_id_indexed_rdd = extracted_uid_bid_rdd \
            .map(lambda record: record[0]) \
            .distinct() \
            .zipWithIndex()
        
        user_id_to_index_map = user_id_indexed_rdd.collectAsMap()

        business_id_indexed_rdd = extracted_uid_bid_rdd \
            .map(lambda record: record[1]) \
            .distinct() \
            .zipWithIndex()
        
        business_id_to_index_map = business_id_indexed_rdd.collectAsMap()
        index_to_bid_map = {v: k for k, v in business_id_to_index_map.items()}

        bid_to_uid_map = extracted_uid_bid_rdd.map(lambda tup: (business_id_to_index_map[tup[1]], user_id_to_index_map[tup[0]])) \
            .groupByKey() \
            .map(lambda tup: (tup[0], list(set(tup[1])))) \
            .collectAsMap()

        N = 70
        buckets = len(user_id_to_index_map)
        # Create Hash Functions
        hash_fns = []
        for i in range(N):
            a = random.randrange(1, 911382371, pow(10, 7) + 100 + 10 + 1)
            b = random.randrange(1, 972667571, pow(10, 7) + 100 + 10 + 1)
            hash_fns.append(self.make_function(a, b, buckets))
        
        user_hashed_indexes_rdd = user_id_indexed_rdd \
            .map(lambda uid_ind_tuple: (user_id_to_index_map[uid_ind_tuple[0]], \
                [func(uid_ind_tuple[1]) for func in hash_fns]))
        
        uid_aggr_bid_data = extracted_uid_bid_rdd \
            .map(lambda record: (user_id_to_index_map[record[0]], business_id_to_index_map[record[1]])) \
            .groupByKey() \
            .map(lambda record: (record[0], list(set(record[1]))))
        
        def chooseLeastHashValue(list1, list2):
            return [min(val1, val2) for (val1, val2) in zip(list1, list2)]
        
        # 1. Using Inner Join below
        sig_matrix_rdd = uid_aggr_bid_data.join(user_hashed_indexes_rdd) \
            .map(lambda tup: tup[1])\
            .flatMap(lambda tup: [(bid, tup[1]) for bid in tup[0]]) \
            .reduceByKey(chooseLeastHashValue)
        
        def createBands(hash_list, num_of_bands):
            N = len(hash_list) 
            chunk_lists = []
            number_of_rows = int(ceil(N / num_of_bands))
            for index, start in enumerate(range(0, N, number_of_rows)):
                chunk_lists.append((index, hash(tuple(hash_list[start: start + number_of_rows]))))
            return chunk_lists
        
        num_of_bands = 70
        candidates = sig_matrix_rdd \
            .flatMap(lambda tup: [(chunk, tup[0]) for chunk in createBands(tup[1], num_of_bands)]) \
            .groupByKey() \
            .map(lambda tup: tup[1]) \
            .filter(lambda businesses: len(businesses) > 1) \
            .flatMap(lambda businesses: [business_pair for business_pair in combinations(businesses, 2)])\
            .collect()
        
        def jaccardSim(s1, s2):
            s1 = set(s1)
            s2 = set(s2)
            return len(s1 & s2) / len(s1 | s2)
        
        res = []
        memo = set()
        for candidate in candidates:
            c1 = candidate[0]
            c2 = candidate[1]
            if ((c1, c2) not in memo and (c2, c1) not in memo):
                memo.add((c1, c2))
                sim = jaccardSim(bid_to_uid_map[c1], bid_to_uid_map[c2])
                if sim >= 0.05:
                    res.append({
                        "b1": index_to_bid_map[c1],
                        "b2": index_to_bid_map[c2],
                        "sim": sim
                    })
        with open(output_file_path, 'w+') as of:
            for obj in res:
                of.write(json.dumps(obj) + "\n")

        of.close()


    def main(self):
      input_file_path = sys.argv[1]
      output_file_path = sys.argv[2]
      conf = SparkConf().setAppName("task1").setMaster("local[*]")
      sc = SparkContext(conf=conf).getOrCreate()
      sc.setLogLevel("OFF")
      self.processInput(sc, input_file_path, output_file_path)



if __name__ == "__main__":
    t1 = task1()
    t1.main()