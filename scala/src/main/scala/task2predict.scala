import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.jackson.JsonMethods._
import org.json4s._
import org.json4s.jackson.Serialization

import java.io.{File, PrintWriter}
object task2predict {
    implicit val formats = DefaultFormats
    def main(args: Array[String]): Unit = {
        val inputFile = args(0)
        val modelFile = args(1)
        val outputFile = args(2)
        val business_const = "b"
        val user_const = "u"
        val conf = new SparkConf().setAppName("task2Predict").setMaster("local[*]")
        val sc: SparkContext = new SparkContext(conf)
        sc.setLogLevel("OFF")
        val t1 = System.nanoTime
        val predictionData = sc.textFile(inputFile).map((row) => parse(row))
        val modelData = sc.textFile(modelFile).map((row) => parse(row))
        val userMap = modelData.filter((row) => {
                ((row \ "t").extract[String]) == "u"
            }).map((row) => {
                ((row \ "uid").extract[String], (row \ "words").extract[Set[String]])
            }).collectAsMap()
        val businessMap = modelData.filter((row) => {
                (row \ "t").extract[String] == "b"})
            .map ((row) => {
                ((row \ "bid").extract[String], (row \ "words").extract[Set[String]])
            }).collectAsMap()
        val res = predictionData.map((record) =>{
            ((record \ "user_id").extract[String], (record \ "business_id").extract[String])
        }).collect()
        var writeData = new scala.collection.mutable.ListBuffer[Tuple3[String, String, Double]]()
        for (tup <- res) {
            val userWords = userMap.getOrElse(tup._1, Set[String]())
            val businessWords = businessMap.getOrElse(tup._2, Set[String]())
            val cosineSim = if (!(userWords.size == 0 || businessWords.size == 0)) {
                ((userWords.intersect(businessWords)).size.toDouble / (scala.math.sqrt(userWords.size) * scala.math.sqrt(businessWords.size)))
            } else {
                0.toDouble
            }
            if (cosineSim >= 0.01) {
                writeData += Tuple3(tup._1, tup._2, cosineSim)
            }
        }
        val writer = new PrintWriter(new File(outputFile))
        writeData.foreach((tup) => {
            val user_id = tup._1
            val business_id = tup._2
            val sim = tup._3
            val record = Map("user_id" -> user_id, "business_id" -> business_id, "sim" -> sim)
            writer.write(Serialization.write(record) + "\n")
        })
        writer.close()
    }
}
