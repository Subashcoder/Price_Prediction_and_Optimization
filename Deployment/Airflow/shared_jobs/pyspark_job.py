from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Read CSV") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "1g") \
    .getOrCreate()

# Read the CSV file
df = spark.read.csv("/opt/Data/listings.csv", header=True, inferSchema=True)

# Repartition for better parallelism
df = df.repartition(4)

# Simple transformation
df = df.filter(df['price'].isNotNull())

# Show the result
print('finished...')

spark.stop()
