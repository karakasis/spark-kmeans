import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

val conf = new SparkConf().setAppName("Simple Application")
val sc = new SparkContext(conf)
// Load and parse the data
val data = sc.textFile("/home/christos/Documents/spark-kmeans/data.csv")
val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()
parsedData.collect().foreach(println)
// Cluster the data into two classes using KMeans
val numClusters = 5
val numIterations = 5
val clusters = KMeans.train(parsedData, numClusters, numIterations)
clusters.clusterCenters(0)
clusters.clusterCenters(1)
clusters.clusterCenters(2)
clusters.clusterCenters(3)
clusters.clusterCenters(4)
// Evaluate clustering by computing Within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(parsedData)
println("Within Set Sum of Squared Errors = " + WSSSE)
parsedData.foreach(x=>{println(x(1))})//prints collumn 1//

// Save and load model
//clusters.save(sc, "target/org/apache/spark/KMeansExample/KMeansModel")
//val sameModel = KMeansModel.load(sc, "target/org/apache/spark/KMeansExample/KMeansModel")

