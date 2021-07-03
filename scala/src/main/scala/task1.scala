import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.json4s.jackson.JsonMethods._
import org.json4s._
import org.json4s.jackson.Serialization

import java.io.{File, PrintWriter}

object task1 {
    def main(args: Array[String]): Unit = {
        val conf: SparkConf = new SparkConf().setAppName("Task1").setMaster("local[*]")
        val sc:SparkContext = new SparkContext(conf)
        sc.setLogLevel("ERROR")
        val inputFilePath = args(0)
        val outputFilePath = args(1)

        val extracted_uid_bid_rdd = sc.textFile(inputFilePath)
            .map((row) => parse(row))
            .map((record) => {
                implicit val formats = DefaultFormats
                ((record \ "user_id").extract[String], (record \ "business_id").extract[String])
            })
        val uid_indexed_rdd = extracted_uid_bid_rdd.map((record) => record._1).distinct().zipWithIndex()
        val uid_to_index_map = uid_indexed_rdd.collectAsMap()
        val bid_indexed_rdd = extracted_uid_bid_rdd.map((record) => record._2).distinct().zipWithIndex()
        val bid_to_index_map = bid_indexed_rdd.collectAsMap()
        val index_to_bid_map = bid_to_index_map.map(_.swap)

        val bid_to_uid_map = extracted_uid_bid_rdd.map(tup => (bid_to_index_map.getOrElse(tup._2, throw new RuntimeException("test2")), uid_to_index_map.getOrElse(tup._1, throw new RuntimeException("test"))))
            .groupByKey()
            .map((record) => {
                (record._1, record._2.toSet)
            })
            .collectAsMap()
        val N = 70
        val buckets = uid_to_index_map.size

        val a_values = List.tabulate(N)(i => {
            val r1 = scala.util.Random
            r1.nextInt(911382371)
        })
        val b_values = List.tabulate(N)(i => {
            val r2 = scala.util.Random
            r2.nextInt(972667571)
        })
        val uid_hashed_indexed_rdd = uid_indexed_rdd.map((record) => {
            val r = scala.util.Random
            val hash = List.tabulate(N)(i => (((a_values(i) * record._2 + b_values(i)) % 1000000009) % buckets))
            (uid_to_index_map.getOrElse(record._1, 0L), hash)
        })
        val uid_aggr_bid_data = extracted_uid_bid_rdd
            .map(record => (uid_to_index_map.getOrElse(record._1, 0L), bid_to_index_map.getOrElse(record._2, 0L)))
            .groupByKey()
            .map((record) => {
                (record._1, record._2.toSet.toList)
            })
        def chooseLeastHashValue(list1: List[Long], list2: List[Long]) = {
            val list3 = for ((val1, val2) <- (list1 zip list2)) yield Math.min(val1, val2)
            list3
        }
        val sig_matrix_rdd = uid_aggr_bid_data.join(uid_hashed_indexed_rdd)
            .map(record => record._2)
            .flatMap(tup => {
                val n = tup._1.size
                List.tabulate(n)(i => (tup._1(i), tup._2))
            })
            .reduceByKey(chooseLeastHashValue).collect()

        val numBands = 70
        val bkt = collection.mutable.Map[Int, scala.collection.immutable.Set[Long]]()
        val candidates = collection.mutable.Set[List[Long]]()
        for (band_num  <- 0 until numBands - 1) {
            bkt.clear()
            val tup_hashCode_uid = sig_matrix_rdd.map((tup_uid_list_hash) => {
                val uid = tup_uid_list_hash._1
                val hashArr = tup_uid_list_hash._2
                val hashCode = hashArr(band_num)
                (hashCode, uid)
            })
            val set_of_uids = tup_hashCode_uid.groupBy(_._1).mapValues(_.map(_._2).toSet)
            for (uids <- set_of_uids.values) {
                if (uids.size > 1) {
                    for (uid_pair <- uids.subsets(2).map(_.toList)) {
                        candidates += uid_pair.sorted
                    }
                }
            }
        }
        def jaccardSim(set_1: Set[Long], set_2: Set[Long]) = {
            (set_1.intersect(set_2)).size / (set_1.union(set_2)).size.toFloat
        }
        val writer = new PrintWriter(new File(outputFilePath))
        for (candidate <- candidates) {
            val c1 = candidate(0)
            val c2 = candidate(1)

                val p1 = bid_to_uid_map(c1)
                val p2 = bid_to_uid_map(c2)
                val sim = jaccardSim(p1, p2)
                if (sim >= 0.05) {
                    implicit val formats = DefaultFormats
                    writer.write(Serialization.write(Map("b1" -> index_to_bid_map.getOrElse(c1, ""), "b2" -> index_to_bid_map.getOrElse(c2, ""), "sim" -> sim)) + "\n")
                }
        }
        writer.close()
    }
}
