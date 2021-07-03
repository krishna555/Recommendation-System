import org.json4s.DefaultFormats
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.jackson.JsonMethods._
import org.json4s._
import org.json4s.jackson.Serialization

import java.io.{File, PrintWriter}

object task3predict {
    implicit val formats = DefaultFormats
    def itemPredict(sc: SparkContext, trainFile: String, testFile: String, modelFile: String, outputFile: String) = {
        val trainRdd = sc.textFile(trainFile).map((row) => parse(row)).map((record) => {
            (((record \ "user_id").extract[String], (record \ "business_id").extract[String]), (record \ "stars").extract[Double])
        })
        val extractedTestData = sc.textFile(testFile).map((row) => parse(row)).map((record) => {
            ((record \ "user_id").extract[String], (record \ "business_id").extract[String])
        })
        val extracted_model_data = sc.textFile(modelFile).map((row) => parse(row)).map((record) => {
            (((record \ "b1").extract[String], (record \ "b2").extract[String]), (record \ "sim").extract[Double])
        }).collect()

        val extractedTrainData = trainRdd
            .groupByKey()
            .mapValues((stars) =>  (stars.sum / stars.size))
            .map((record) => (record._1._1, (record._1._2, record._2)))
            .groupByKey()
            .mapValues((value) => value.toSet.toList)

        var model = new scala.collection.mutable.HashMap[Tuple2[String, String], Double]()
        extracted_model_data.map((record) => {
            val business_pair = record._1
            val bid1 = business_pair._1
            val bid2 = business_pair._2
            val sortedItems = Array(bid1, bid2).sorted
            val sortedTuple = (sortedItems(0), sortedItems(1))
            val sim = record._2
            model(sortedTuple) = sim
        })
        val predictions = extractedTrainData.join(extractedTestData)
            .map((tup) => {
                val user_id = tup._1
                val business_id = tup._2._2
                val bid_stars_tuple_list = tup._2._1
                val filteredBusinesses = bid_stars_tuple_list.filter(tup => {
                    tup._1 != business_id
                })
                ((user_id, business_id), filteredBusinesses)
            })
            .groupByKey()
            .flatMapValues((value) => value.toList)
            .map((record) => {
                val uid = record._1._1
                val bid = record._1._2
                var neighbours = scala.collection.mutable.ListBuffer[Tuple2[Double, Double]]()
                record._2.map((record) => {
                    val bid2 = record._1
                    val rating = record._2
                    val sortedItems = Array(bid, bid2).sorted
                    val sortedTuple = (sortedItems(0), sortedItems(1))
                    if (model.contains(sortedTuple)) {
                        neighbours += Tuple2(rating, model(sortedTuple))
                    }
                })
                val topN = neighbours.sortBy((x) => -x._2).take(10)
                var num_sum = 0.toDouble
                var den_sum = 0.toDouble
                topN.foreach((rating_sim_pair) => {
                    num_sum += rating_sim_pair._1 * rating_sim_pair._2
                    den_sum += math.abs(rating_sim_pair._2)
                })
                val stars = if (num_sum != 0 && den_sum != 0) {
                    num_sum / den_sum
                }  else 0
                if (stars > 0)
                    (uid, bid, stars)
                else
                    (uid, bid, 0)
            }).filter((record) => record._3 != 0).collect()

        val writer = new PrintWriter(new File(outputFile))
        predictions.foreach((record) => {
            val userId = record._1
            val businessId = record._2
            val stars = record._3
            val outputData = Map("user_id" -> userId, "business_id" -> businessId, "stars" -> stars)
            writer.write(Serialization.write(outputData) + "\n")
        })
        writer.close()
    }

    def userPredict(sc: SparkContext, trainFile: String, testFile: String, modelFile: String, outputFile: String) = {
        val extractedTrainData = sc.textFile(trainFile).map((row) => parse(row)).map((record) => {
            ((record \ "business_id").extract[String], ((record \ "user_id").extract[String], (record \ "stars").extract[Double]))
        })
        val extractedTestData = sc.textFile(testFile).map((row) => parse(row)).map((record) => {
            ((record \ "business_id").extract[String], (record \ "user_id").extract[String])
        })
        val extracted_model_data = sc.textFile(modelFile).map((row) => parse(row)).map((record) => {
            (((record \ "u1").extract[String], (record \ "u2").extract[String]), (record \ "sim").extract[Double])
        }).collect()

        val bid_aggr_train_data = extractedTrainData.groupByKey().mapValues((user_stars_tuples) => user_stars_tuples.toSet.toList)

        val uid_tup_bid_rating = extractedTrainData.map((record) => (record._2._1, (record._1, record._2._2)))
            .groupByKey()
            .map(record => (record._1, record._2.toList)).collect()

        var uid_to_sum_cnt_map = new scala.collection.mutable.HashMap[String, Tuple2[Double, Int]]
        uid_tup_bid_rating.foreach((tup_uid_tup_bid_rating) => {
            val uid = tup_uid_tup_bid_rating._1
            val bid_ratings = tup_uid_tup_bid_rating._2
            val num_ratings = bid_ratings.size
            var sum_ratings = 0.toDouble
            bid_ratings.foreach((ratings) => {
                val rating = ratings._2
                sum_ratings += rating
            })
            uid_to_sum_cnt_map(uid) = Tuple2(sum_ratings, num_ratings)
        })
        var model = new scala.collection.mutable.HashMap[Tuple2[String, String], Double]()
        extracted_model_data.map((record) => {
            val uid_pair = record._1
            val uid1 = uid_pair._1
            val uid2 = uid_pair._2
            val sortedItems = Array(uid1, uid2).sorted
            val sortedTuple = (sortedItems(0), sortedItems(1))
            val sim = record._2
            model(sortedTuple) = sim
        })

        val predictions = bid_aggr_train_data.join(extractedTestData)
            .map((tup) => {
                val business_id = tup._1
                val user_id = tup._2._2
                val uid_stars_tuple_list = tup._2._1
                val filteredUsers = uid_stars_tuple_list.filter(tup => {
                    tup._1 != user_id
                })
                ((business_id, user_id), filteredUsers)
            })
            .groupByKey()
            .flatMapValues((value) => value.toList)
            .map((record) => {
                val bid = record._1._1
                val uid1 = record._1._2
                var neighbours = scala.collection.mutable.ListBuffer[Tuple3[Double, Double, String]]()
                record._2.map((uid_stars_tup) => {
                    val uid2 = uid_stars_tup._1
                    val rating = uid_stars_tup._2
                    val sortedItems = Array(uid1, uid2).sorted
                    val sortedTuple = (sortedItems(0), sortedItems(1))
                    if (model.contains(sortedTuple)) {
                        neighbours += Tuple3(rating, model(sortedTuple), uid2)
                    }
                })
                val topN = neighbours.sortBy((x) => -x._2).take(10)
                var num_sum = 0.toDouble
                var den_sum = 0.toDouble

                topN.foreach((rating_sum_uid_tup) => {
                    val rtg = rating_sum_uid_tup._1
                    val sim = rating_sum_uid_tup._2
                    val uid = rating_sum_uid_tup._3
                    val sum_and_cnt_tup = uid_to_sum_cnt_map(uid)
                    val uid_sum = sum_and_cnt_tup._1
                    val uid_cnt = sum_and_cnt_tup._2
                    val avg_term = if (uid_cnt - 1 > 0)  {
                        (uid_sum - rtg) / (uid_cnt - 1)
                    } else {
                        0
                    }
                    val num_curr = (rtg - avg_term) * sim
                    num_sum += num_curr
                    den_sum += math.abs(sim)
                })

                val term2 = if (num_sum != 0 && den_sum != 0) {
                    num_sum / den_sum
                } else {
                    0
                }
                val uid1_sum = uid_to_sum_cnt_map(uid1)._1
                val uid1_cnt = uid_to_sum_cnt_map(uid1)._2
                val avg_term_curr = uid1_sum / uid1_cnt
                (uid1, bid, avg_term_curr + term2)
            })
            .filter((tup) => tup._3 != 0).collect()

        val writer = new PrintWriter(new File(outputFile))
        predictions.foreach((record) => {
            val userId = record._1
            val businessId = record._2
            val stars = record._3
            val outputData = Map("user_id" -> userId, "business_id" -> businessId, "stars" -> stars)
            writer.write(Serialization.write(outputData) + "\n")
        })
        writer.close()
    }
    def main(args: Array[String]) = {
        val conf: SparkConf = new SparkConf().setAppName("task3").setMaster("local[*]")
        val sc: SparkContext = new SparkContext(conf)
        sc.setLogLevel("OFF")
        val trainFile = args(0)
        val testFile = args(1)
        val modelFile = args(2)
        val outputFile = args(3)
        val cfType = args(4)
        if (cfType == "item_based") {
            itemPredict(sc, trainFile, testFile, modelFile, outputFile)
        }
        else {
            userPredict(sc, trainFile, testFile, modelFile, outputFile)
        }
    }
}
