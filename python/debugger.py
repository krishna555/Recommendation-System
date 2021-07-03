from pyspark import SparkConf, SparkContext
import json
import math
conf = SparkConf().setAppName("task3").setMaster("local[*]")
sc = SparkContext(conf=conf).getOrCreate()
expected_rdd = sc.textFile("./resources/test_review_ratings.json").map(lambda row: json.loads(row))

expected = expected_rdd.map(lambda record: (record["user_id"], record["business_id"], record["stars"])).collect()

predicted_map = sc.textFile("./resources/task3user.predict").map(lambda row: json.loads(row)).map(lambda record: ((record["user_id"], record["business_id"]), record["stars"])).collectAsMap()
cnt = 0
ans = 0
max_diff = -math.inf
for tup in expected:
    cnt += 1
    uid, bid, stars = tup
    predicted_stars = 0
    if (uid, bid) in predicted_map:
        predicted_stars = predicted_map[(uid, bid)]
    if abs((predicted_stars - stars)) > max_diff:
        max_diff = abs((predicted_stars - stars))
        print(predicted_stars, stars, uid, bid)
    term = (predicted_stars - stars) * (predicted_stars - stars)
    ans += term
print(math.sqrt(ans / cnt))
print(max_diff)


