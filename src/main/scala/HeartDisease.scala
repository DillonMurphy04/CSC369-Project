import org.apache.spark.SparkContext._
import scala.io._
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection._
object HeartDisease {

  def main(args: Array[String]):Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("HeartDisease")
      .setMaster("local[4]")
    val sc = new SparkContext(conf)

    val all = sc.textFile("heart_2020_cleaned.csv")
    val header = all.first()
    val data = all.take(10).map(_.split(","))

    // quantitative variables for binning
    val columnNames = header.split(",")
    val bmiIdx = columnNames.indexOf("BMI")
    val pHealthIdx = columnNames.indexOf("PhysicalHealth")
    val mHealthIdx = columnNames.indexOf("MentalHealth")
    val sleepTimeIdx = columnNames.indexOf("sleepTimeIdx")
  }
}
