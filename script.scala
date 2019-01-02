import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix

val conf = new SparkConf().setAppName("Simple Application")
val sc = new SparkContext(conf)
// Load and parse the data
val data = sc.textFile("/home/dimitris/Desktop/spark/spark-kmeans/data.csv")
val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
val mat: RowMatrix = new RowMatrix(parsedData)
//parsedData.collect().foreach(println)
// Cluster the data into two classes using KMeans
val numClusters = 5
val numIterations = 20
val clusters = KMeans.train(parsedData, numClusters, numIterations)

println("@@@@@@@@@@@@@@@")
println("Print each point which class belongs")
clusters.predict(parsedData).foreach(println)
println("@@@@@@@@@@@@@@@")
clusters.clusterCenters(0)
clusters.clusterCenters(1)
clusters.clusterCenters(2)
clusters.clusterCenters(3)
clusters.clusterCenters(4)
// Evaluate clustering by computing Within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(parsedData)
println("Within Set Sum of Squared Errors = " + WSSSE)


val summary: MultivariateStatisticalSummary = Statistics.colStats(parsedData)
println(summary.mean)  // a dense vector containing the mean value for each column
println("--------------------------")
println("-----COVARIANCE MATRIX----")
println("--------------------------")
val cov: Matrix = mat.computeCovariance()
//println(summary.numNonzeros)  // number of nonzeros in each column
//parsedData.foreach(x=>{println(x(1))})//prints collumn 1//

// Save and load model
//clusters.save(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
//val sameModel = KMeansModel.load(sc, "target/org/apache/spark/KMeansExample/KMeansModel")

