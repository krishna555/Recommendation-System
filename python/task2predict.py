import sys
from pyspark import SparkConf, SparkContext
import json
import math

if __name__ == "__main__":
    input_file = sys.argv[1]
    model_file = sys.argv[2]
    output_file = sys.argv[3]
    BUSINESS_CONST = "b"
    USER_CONST = "u"
    conf = SparkConf().setAppName("task1").setMaster("local[*]")
    sc = SparkContext(conf=conf).getOrCreate()
    sc.setLogLevel("OFF")

    prediction_data = sc.textFile(input_file).map(lambda row: json.loads(row))
    model_data = sc.textFile(model_file).map(lambda row: json.loads(row))
    user_map = model_data.filter(lambda row: row["t"] == "u").map(lambda row: (row['uid'], row['words'])).collectAsMap()
    business_map = model_data.filter(lambda row: row["t"] == "b").map(lambda row: (row['bid'], row['words'])).collectAsMap()

    def cosine_similarity(uwords, bwords):

        if uwords == None or bwords == None:
            return 0
        else:
            a = set(uwords)
            b = set(bwords)
            return len(a & b) / (math.sqrt(len(a)) * math.sqrt(len(b)))
    
    
    res = prediction_data.map(lambda record: (record["user_id"], record["business_id"]))\
        .map(lambda row: (row[0], row[1], cosine_similarity(user_map.get(row[0]), business_map.get(row[1]))))\
        .filter(lambda row: row[2] >= 0.01).collect()
    
    with open(output_file, 'w+') as of:
        for tup in res:
            obj = {}
            obj['user_id'] = tup[0]
            obj['business_id'] = tup[1]
            obj['sim'] = tup[2]
            of.write(json.dumps(obj) + "\n")
    of.close()
