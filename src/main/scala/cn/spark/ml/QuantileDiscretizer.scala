package cn.spark.ml

import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.sql.SparkSession

/**
  * Created by 张宝玉 on 2018/8/10.
  */
object QuantileDiscretizer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("").getOrCreate()
    val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
    val df = spark.createDataFrame(data).toDF("id", "hour")

    val discretizer = new QuantileDiscretizer()
      .setInputCol("hour")
      .setOutputCol("result")
      .setNumBuckets(3)

    val result = discretizer.fit(df).transform(df)
    result.show()
    spark.stop()
  }
}
