import org.json4s.DefaultFormats
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.jackson.JsonMethods._
import org.json4s._
import org.json4s.jackson.Serialization

import java.io.{File, PrintWriter}
import scala.collection.mutable

object task3train {
    implicit val formats = DefaultFormats
    def get_pearson_sim(uid_rtg_map1: scala.collection.mutable.Map[Long, Double], uid_rtg_map2: scala.collection.mutable.Map[Long, Double], uid1Keys: Set[Long], uid2Keys: Set[Long]): Double = {
        val corated_items = uid1Keys.intersect(uid2Keys)

        var corated_items_1_sum = 0.toDouble
        var corated_items_1_cnt = 0.toDouble
        var corated_items_2_sum = 0.toDouble
        var corated_items_2_cnt = 0.toDouble
        val corated_items_zipped = corated_items.map((item) => {
            corated_items_1_sum += uid_rtg_map1(item)
            corated_items_1_cnt += 1
            corated_items_2_sum += uid_rtg_map2(item)
            corated_items_2_cnt += 1
            (uid_rtg_map1(item), uid_rtg_map2(item))
        })

        val avg1 = corated_items_1_sum / corated_items_1_cnt
        val avg2 = corated_items_2_sum / corated_items_2_cnt

        var num_sum = 0.toDouble
        var den_sum1 = 0.toDouble
        var den_sum2 = 0.toDouble
        for (tup <- corated_items_zipped) {
            val rtg1 = tup._1
            val rtg2 = tup._2
            val new_rtg1 = rtg1 - avg1
            val new_rtg2 = rtg2 - avg2
            num_sum = num_sum  +  (new_rtg1 * new_rtg2)
            den_sum1 += math.pow(new_rtg1, 2)
            den_sum2 += math.pow(new_rtg2, 2)
        }
        val den_sum = math.sqrt(den_sum1) * math.sqrt(den_sum2)
        if (num_sum > 0 && den_sum > 0) {
            return num_sum.toDouble / den_sum
        }
        else {
            return 0.toDouble
        }
    }
    def itemTrain(sc: SparkContext, trainInputPath: String, outputFilePath: String) = {
        val t1 = System.nanoTime
        val extracted_train_rdd = sc.textFile(trainInputPath).map((row) => parse(row)).map((record) => {
            ((record \ "user_id").extract[String], ((record \ "business_id").extract[String], (record \ "stars").extract[Double]))
        })
        val user_id_indexed_rdd = extracted_train_rdd.map((record) => record._1).distinct().zipWithIndex()
        val user_id_to_index_map = user_id_indexed_rdd.collectAsMap()
        val business_id_indexed_rdd =    extracted_train_rdd.map((record) => record._2._1).distinct().zipWithIndex()
        val business_id_to_index_map = business_id_indexed_rdd.collectAsMap()
        val index_to_bid_map = business_id_to_index_map.map(_.swap)

        val bid_tup_uid_rating = extracted_train_rdd.map((record) => (business_id_to_index_map(record._2._1), (user_id_to_index_map(record._1), record._2._2)))
            .groupByKey()
            .map((record) => (record._1, record._2.toList))
            .filter(tup => tup._2.size >= 3).collect()

        var bid_uid_rtg_map = new scala.collection.mutable.HashMap[Long, scala.collection.mutable.HashMap[Long, Double]]()
        bid_tup_uid_rating.foreach((tup_bid_tup_uid_rating) => {
                val bid = tup_bid_tup_uid_rating._1
                val uid_ratings = tup_bid_tup_uid_rating._2
                uid_ratings.foreach((ratings) => {
                    val uid = ratings._1
                    if (!bid_uid_rtg_map.contains(bid)) {
                        bid_uid_rtg_map(bid) = new scala.collection.mutable.HashMap[Long, Double]
                    }
                    bid_uid_rtg_map(bid)(uid) = ratings._2
                })
            })
        val writer = new PrintWriter(new File(outputFilePath))
        val bid_pairs = bid_uid_rtg_map.keys.toSet.subsets(2).map(_.toList)

        // Key Set Map:
        val keyGenMap = bid_uid_rtg_map.map((tup) => {
            tup._1 -> tup._2.keys.toSet
        })
        for (bid_pair <- bid_pairs) {
            val bid1 = bid_pair(0)
            val bid2 = bid_pair(1)
            val uid1 = bid_uid_rtg_map(bid1)
            val uid2 = bid_uid_rtg_map(bid2)

                val uid1_keys = keyGenMap(bid1)
                val uid2_keys = keyGenMap(bid2)
                val similarities = uid1_keys.intersect(uid2_keys)
                if (similarities.size >= 3) {
                    val ps = get_pearson_sim(uid1, uid2, uid1_keys, uid2_keys)
                    if (ps > 0) {
                        val writeObj = Map("b1" -> index_to_bid_map(bid1), "b2" -> index_to_bid_map(bid2), "sim" -> ps)
                        writer.write(Serialization.write(writeObj) + "\n")
                    }
                }
            }
        writer.close()
    }

    def userTrain(sc: SparkContext,  trainInputPath: String, outputFilePath: String) = {
        val t1 = System.nanoTime()
        val extracted_train_rdd = sc.textFile(trainInputPath).map((row) => parse(row)).map((record) => {
            ((record \ "user_id").extract[String], ((record \ "business_id").extract[String], (record \ "stars").extract[Double]))
        })
        val user_id_indexed_rdd = extracted_train_rdd.map((record) => record._1).distinct().zipWithIndex()
        val user_id_to_index_map = user_id_indexed_rdd.collectAsMap()
        val index_to_uid_map = user_id_to_index_map.map(_.swap)
        val business_id_indexed_rdd = extracted_train_rdd.map((record) => record._2._1).distinct().zipWithIndex()
        val business_id_to_index_map = business_id_indexed_rdd.collectAsMap()
        val uid_tup_bid_rating = extracted_train_rdd.map((record) => (user_id_to_index_map(record._1), (business_id_to_index_map(record._2._1), record._2._2)))
                .groupByKey()
                .map((record) => (record._1, record._2.toList))
                .filter((record) => record._2.size >= 3)
        val uid_tup_bid_rating_collected = uid_tup_bid_rating.collect()

        var uid_bid_rtg_map = new scala.collection.mutable.HashMap[Long, scala.collection.mutable.HashMap[Long, Double]]()

        uid_tup_bid_rating_collected.foreach((tup_uid_tup_bid_rating) => {
            val uid = tup_uid_tup_bid_rating._1
            val bid_ratings = tup_uid_tup_bid_rating._2
            bid_ratings.foreach((bid_rating) => {
                val bid = bid_rating._1
                val rating = bid_rating._2
                if (!uid_bid_rtg_map.contains(uid)) {
                    uid_bid_rtg_map(uid) = new scala.collection.mutable.HashMap[Long, Double]
                }
                uid_bid_rtg_map(uid)(bid) = rating
            })
        })
        val tup_bid_uid_rdd = uid_tup_bid_rating.flatMap((record) => {
            record._2.map((bid_rating) => {
                (bid_rating._1, record._1)
            })
        })
        val t2 = System.nanoTime
        val N = 70
        val a_values = List.tabulate(N)(i => {
            val r1 = scala.util.Random
            r1.nextInt(911382371)
        })
        val b_values = List.tabulate(N)(i => {
            val r2 = scala.util.Random
            r2.nextInt(972667571)
        })
        val buckets = user_id_to_index_map.size
        val bid_to_hash = business_id_to_index_map.map((tup) => {
            val hash = List.tabulate(N)(i => (((a_values(i) * tup._2 + b_values(i)) % 1000000009) % buckets))
            (tup._2, hash)
        })
        def chooseLeastHashValue(list1: List[Long], list2: List[Long]) = {
            val list3 = for ((val1, val2) <- (list1 zip list2)) yield Math.min(val1, val2)
            list3
        }
        val sig_matrix_uid_n_hash_list = tup_bid_uid_rdd.map((record) => {
            (record._1, record._2)
        }).groupByKey().mapValues((record) => record.toSet.toList)
            .flatMap((bid_uid_list) => {
                bid_uid_list._2.map((user_id) => {
                    (user_id, bid_to_hash(bid_uid_list._1))
                })
            }).reduceByKey(chooseLeastHashValue).collect()
        val t3 = System.nanoTime
        val numBands = 70
        val bkt = collection.mutable.Map[Int, scala.collection.immutable.Set[Long]]()
        val candidates = collection.mutable.Set[List[Long]]()
        for (band_num  <- 0 until numBands - 1) {
            bkt.clear()
            val tup_hashCode_uid = sig_matrix_uid_n_hash_list.map((tup_uid_list_hash) => {
                val uid = tup_uid_list_hash._1
                val hashArr = tup_uid_list_hash._2
                val hashCode = hashArr(band_num)
                (hashCode, uid)
            })
            val set_of_uids = tup_hashCode_uid.groupBy(_._1).mapValues(_.map(_._2).toSet)
            for (uids <- set_of_uids.values) {
                if (uids.size > 1) {
                    for (uid_pair <- uids.toSet.subsets(2).map(_.toList)) {
                        candidates += uid_pair.sorted
                    }
                }
            }
        }
        def jaccardSim(set_1: Set[Long], set_2: Set[Long]) = {
            (set_1.intersect(set_2)).size / (set_1.union(set_2)).size.toFloat
        }

        // Generate Key Set Logic:
        val keySetMap = uid_bid_rtg_map.map((tup) => {
            tup._1 -> tup._2.keys.toSet
        })
        val writer = new PrintWriter(new File(outputFilePath))
        for (candidate <- candidates) {
            val user1 = candidate(0)
            val user2 = candidate(1)

            val b1 = uid_bid_rtg_map(user1)
            val b2 = uid_bid_rtg_map(user2)
            val b1_keys = keySetMap(user1)
            val b2_keys = keySetMap(user2)
            if ((b1_keys.intersect(b2_keys)).size >= 3) {
                val sim = jaccardSim(b1_keys, b2_keys)
                if (sim >= 0.01) {
                    val ps = get_pearson_sim(b1, b2, b1_keys, b2_keys)
                    if (ps > 0) {
                        writer.write(Serialization.write(Map("u1" -> index_to_uid_map(user1), "u2" -> index_to_uid_map(user2), "sim" -> ps)) + "\n")
                    }
                }
            }

        }

        writer.close()
    }
    def main(args: Array[String]): Unit = {
        val conf: SparkConf = new SparkConf().setAppName("task2").setMaster("local[*]")
        val sc: SparkContext = new SparkContext(conf)
        sc.setLogLevel("OFF")
        val trainInputPath = args(0)
        val outputFilePath = args(1)
        val cf_type = args(2)
        if (cf_type == "item_based") {
            itemTrain(sc, trainInputPath, outputFilePath)
        }
        else {
            userTrain(sc, trainInputPath, outputFilePath)
        }
        sc.stop()
    }
}
