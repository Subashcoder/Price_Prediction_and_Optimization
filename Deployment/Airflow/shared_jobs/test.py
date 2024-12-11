from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ResourceAllocationTest").getOrCreate()

# Generate a large RDD with numbers
data = spark.sparkContext.parallelize(range(1, 10**8))  # Adjust the size if needed

# Perform a computation (sum all numbers)
result = data.sum()

# Print the result
print(f"Sum of numbers from 1 to {10**8 - 1} is {result}")

# Stop Spark session
spark.stop()