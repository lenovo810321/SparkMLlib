package cn.spark.ml

import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors

/**
  * Created by 张宝玉 on 2018/8/10.
  */
object PolynomialExpansion {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("").getOrCreate()
    val data = Array(
      Vectors.dense(2.0, 1.0),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(3.0, -1.0)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    df.show()

    val polyExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(3)

    val polyDF = polyExpansion.transform(df)
    polyDF.show(false)

    spark.stop()
  }
}
