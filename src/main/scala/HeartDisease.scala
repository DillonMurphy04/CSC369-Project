import breeze.numerics.sqrt
import scala.io.Source
import scala.math.log
import scala.io.Source
import scala.util.Random
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.PriorityQueue
import org.apache.log4j.{Level, Logger}
import scala.math.exp
import scala.util.Random
case class Point(features: List[Double], label: Double)

object HeartDisease {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val rdd = Source.fromFile("heart_disease.csv").getLines.toList

    val n = 5000

    val Y = rdd
      .map(_.split(",")(0))
      .map(_.toDouble)
      .take(n)

    val X: List[List[Double]] = rdd
      .map(_.split(",", 2)(1))
      .map(_.split(",").map(_.toDouble).toList)
      .take(n)


    //Split data into training and testing sets
    val splitRatio = 0.8
    val splitIndex = (X.length * splitRatio).toInt
    val (trainX, testX) = X.splitAt(splitIndex)
    val (trainY, testY) = Y.splitAt(splitIndex)

    val theta = logisticRegression(trainX, trainY)

    println(s"Optimal parameters: $theta")

    val predictions = predict(testX, theta)
    println(predictions)

    val macroF1 = macroF1Score(predictions, testY)
    println(macroF1)
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

  def sigmoid(z: Double): Double = {
    1.0 / (1.0 + exp(-z))
  }

  def hypothesis(theta: List[Double], x: List[Double]): Double = {
    sigmoid((theta zip x).map { case (a, b) => a * b }.sum)
  }

  def logisticRegression(X: List[List[Double]], y: List[Double], alpha: Double = 0.01, iterations: Int = 1000): List[Double] = {
    val m = X.length
    val n = X.head.length

    // Initialize theta with random values
    var theta = List.fill(n)(Random.nextDouble())

    // Perform iterations without explicit for loops
    (1 to iterations).foldLeft(theta) { (theta, _) =>
      val h = X.map(hypothesis(theta, _))
      val error = h.zip(y).map { case (h_i, y_i) => h_i - y_i }
      val gradient = (0 until n).map { j =>
        (0 until m).map { i =>
          error(i) * X(i)(j)
        }.sum
      }.toList
      theta.zip(gradient).map { case (theta_j, gradient_j) =>
        theta_j - alpha * gradient_j / m.toDouble
      }
    }
  }

  def predict(X_test: List[List[Double]], theta: List[Double]): List[Double] = {
    X_test.map(features => if (hypothesis(theta, features) >= 0.5) 1.0 else 0.0)
  }

//  def euclideanDistance(vector1: List[Double], vector2: List[Double]): Double = {
//    require(vector1.length == vector2.length, "Vectors must have the same length")
//
//    val squaredDistances = vector1.zip(vector2).map { case (x1, x2) =>
//      val diff = x1 - x2
//      diff * diff
//    }
//
//    math.sqrt(squaredDistances.sum)
//  }
//
//
//  def knn(X: List[List[Double]], Y: List[Double], k: Int, points: List[List[Double]]): List[Double] = {
//    val allPoints = X.zip(Y).map { case (features, label) => Point(features, label) }
//
//    points.map { newPoint =>
//      val nearestNeighbors = PriorityQueue.empty[(Point, Double)](Ordering.by(_._2))
//      allPoints.foreach { p =>
//        val distance = euclideanDistance(p.features, newPoint)
//        nearestNeighbors.enqueue((p, distance))
//        if (nearestNeighbors.size > k) nearestNeighbors.dequeue()
//      }
//
//      val labelCounts = nearestNeighbors.map(_._1.label).groupBy(identity).mapValues(_.size)
//      labelCounts.maxBy(_._2)._1
//    }
//  }
//
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

}
