import scala.io.Source
import scala.collection.mutable.PriorityQueue
import org.apache.log4j.{Level, Logger}
import scala.math._
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation



case class Point(features: List[Double], label: Double)

object HeartDisease {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val rdd = Source.fromFile("heart_disease.csv").getLines.toList
    val n = 300000

    val Y = rdd
      .map(_.split(",")(0))
      .map(_.toDouble)
      .take(n)

    val X: List[List[Double]] = rdd
      .map(_.split(",", 2)(1))
      .map(_.split(",").map(_.toDouble).toList)
      .take(n)


    val corrs = correlationList(X, Y)

    val biggestCorrs = corrs.zipWithIndex.sortBy(-_._1).take(13).map(_._2)
    val X_sub = X.map(innerList => biggestCorrs.map(index => innerList(index)))

    // Split data into training and testing sets
    val splitRatio = 0.999
    val splitIndex = (X.length * splitRatio).toInt
    val (trainX, testX) = X_sub.splitAt(splitIndex)
    val (trainY, testY) = Y.splitAt(splitIndex)

    // Run KNN
    val predictions = knn(trainX, trainY, 6, testX)
    val macroF1 = macroF1Score(predictions, testY)
    println(macroF1)
    val matrix = confusionMatrix(predictions, testY)
    println("Confusion Matrix:")
    matrix.foreach(row => println(row.mkString("\t")))


    //    // Run KNN
    //    val minK = 2 //change to test different min values of k
    //    val maxK = 30 //change to test different max values of k
    //    val kValues = (minK to maxK by 2).toList
    //
    //    val result = kValues.map{ k =>
    //      val preds = knn(trainX, trainY, k, testX)
    //      val macroF1 = macroF1Score(preds, testY)
    //      (k, macroF1)
    //    }.sortBy(-_._2)
    //
    //    result.foreach(println)
  }



  def euclideanDistance(vector1: List[Double], vector2: List[Double]): Double = {
    require(vector1.length == vector2.length, "Vectors must have the same length")

    val squaredDistances = vector1.zip(vector2).map { case (x1, x2) =>
      val diff = x1 - x2
      diff * diff
    }

    math.sqrt(squaredDistances.sum)
  }


  def knn(X: List[List[Double]], Y: List[Double], k: Int, points: List[List[Double]]): List[Double] = {
    val allPoints = X.zip(Y).map { case (features, label) => Point(features, label) }

    points.map { newPoint =>
      val nearestNeighbors = PriorityQueue.empty[(Point, Double)](Ordering.by(_._2))
      allPoints.foreach { p =>
        val distance = euclideanDistance(p.features, newPoint)
        nearestNeighbors.enqueue((p, distance))
        if (nearestNeighbors.size > k) nearestNeighbors.dequeue()
      }

      val labelCounts = nearestNeighbors.map(_._1.label).groupBy(identity).mapValues(_.size)
      labelCounts.maxBy(_._2)._1
    }
  }

  def macroF1Score(predictions: List[Double], labels: List[Double]): Double = {
    require(predictions.length == labels.length, "Number of predictions must be equal to the number of labels")

    val uniqueLabels = (predictions ++ labels).distinct
    val f1Scores = uniqueLabels.map { label =>
      val truePositives = predictions.zip(labels).count { case (predicted, actual) =>
        predicted == label && actual == label
      }
      val falsePositives = predictions.zip(labels).count { case (predicted, actual) =>
        predicted == label && actual != label
      }
      val falseNegatives = predictions.zip(labels).count { case (predicted, actual) =>
        predicted != label && actual == label
      }
      val precision = truePositives.toDouble / (truePositives + falsePositives)
      val recall = truePositives.toDouble / (truePositives + falseNegatives)
      val f1 = 2 * (precision * recall) / (precision + recall)
      if (f1.isNaN) 0.0 else f1
    }

    f1Scores.sum / f1Scores.length
  }

  def confusionMatrix(predictions: List[Double], testY: List[Double]): Array[Array[Int]] = {
    val tp = predictions.zip(testY).count { case (pred, actual) => pred == 1.0 && actual == 1.0 }
    val fp = predictions.zip(testY).count { case (pred, actual) => pred == 1.0 && actual == 0.0 }
    val tn = predictions.zip(testY).count { case (pred, actual) => pred == 0.0 && actual == 0.0 }
    val fn = predictions.zip(testY).count { case (pred, actual) => pred == 0.0 && actual == 1.0 }

    Array(Array(tp, fp), Array(fn, tn))
  }

  def correlationList(x: List[List[Double]], y: List[Double]): List[Double] = {
    val correlations = new PearsonsCorrelation()
    val transposedX = x.transpose
    transposedX.map(feature => correlations.correlation(feature.toArray, y.toArray))
  }

}
