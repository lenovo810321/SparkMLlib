package cn.spark.ml

import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.sql.SparkSession

/**
  * Created by 张宝玉 on 2018/8/10.
  *
  * SQL转换器
  */
object SQLTransform {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("").getOrCreate()
    val df = spark.createDataFrame(Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")
    val sqlTrans = new SQLTransformer().setStatement(
      "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__"
    )
    sqlTrans.transform(df).show()

    spark.stop()
  }
}
