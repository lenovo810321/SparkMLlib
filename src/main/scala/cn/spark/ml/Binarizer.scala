package cn.spark.ml

import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.sql.SparkSession

/**
  * Created by 张宝玉 on 2018/8/10.
  */
object Binarizer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("Binarizer")getOrCreate()

    val data = Array((0,0.1),(1,0.8),(2,0.2))
    val dataFrame = spark.createDataFrame(data).toDF("id","feature")

    val binarizer: Binarizer = new Binarizer()
      .setInputCol("feature")
      .setOutputCol("binarized_feature")
      .setThreshold(0.5)

    val binarizerDataFrame = binarizer.transform(dataFrame)
    println(s"Bnarizer output with Threshold = ${binarizer.getThreshold}")
    binarizerDataFrame.show()

    spark.stop()
  }
}
