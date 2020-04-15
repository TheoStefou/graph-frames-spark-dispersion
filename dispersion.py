from graphframes import GraphFrame
from graphframes.lib import AggregateMessages as AM
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, functions as sqlfunctions, types
from pyspark.sql import functions as F

from itertools import combinations

spark = SparkSession.builder.appName('my-network').getOrCreate()

vertices = spark.read.format('csv').option('header', 'true').load('./vertices.csv')
edges = spark.read.format('csv').option('header', 'true').load('./edges.csv')

#---------------------------------------
print('State 1')
vertices.show()
edges.show()

#---------------------------------------
print('State 2')
neighbors_of_neighbors_type = types.ArrayType(types.StructType([types.StructField('neighbor_id', types.StringType()), types.StructField('neighbor_neighbors', types.ArrayType(types.StringType()))]))
neighbors_type = types.StructType([types.StructField('neighbor_id', types.StringType()), types.StructField('neighbor_neighbors', types.ArrayType(types.StringType()))])
grouped_edges = edges.groupBy('src').agg(F.collect_set('dst').alias('neighbors')).withColumnRenamed('src', 'id')

def create_id_list_column(id, neighbors):
  return {'neighbor_id': id , 'neighbor_neighbors': neighbors}
create_id_list_column_udf = F.udf(create_id_list_column, neighbors_type)

vertices = grouped_edges.withColumn('neighbors_of_neighbors', create_id_list_column_udf(grouped_edges['id'], grouped_edges['neighbors'])).join(vertices, on='id', how='left_outer').drop('neighbors')
vertices.show(500, truncate=False)

#---------------------------------------
print('State 3')
g = GraphFrame(vertices, edges)
#g.vertices.show(500, truncate=False)
#g.edges.show(500, truncate=False)

aggregates = g.aggregateMessages(F.collect_set(AM.msg).alias('neighbors_of_neighbors2'),
              sendToDst=AM.src['neighbors_of_neighbors'])

vertices = vertices.join(aggregates, on='id', how='left_outer').withColumnRenamed('neighbors_of_neighbors', 'neighbors').withColumnRenamed('neighbors_of_neighbors2', 'neighbors_of_neighbors')
vertices.show(500, truncate=False)

#---------------------------------------
print('State 4')
def same_neighbors(neighbors, neighbors_of_neighbors):
  my_neighbors = neighbors['neighbor_neighbors']
  retL = []
  for item in neighbors_of_neighbors:
    dic = {'neighbor_id': item['neighbor_id']}
    same_neigh = []
    for neigh in item['neighbor_neighbors']:
      if neigh in my_neighbors:
        same_neigh.append(neigh)
    dic['neighbor_neighbors'] = same_neigh
    retL.append(dic)
  return retL
same_neighbors_udf = F.udf(same_neighbors, neighbors_of_neighbors_type)

vertices = vertices.withColumn('same_neighbors', same_neighbors_udf(vertices['neighbors'], vertices['neighbors_of_neighbors'])).drop('neighbors').drop('neighbors_of_neighbors')

vertices.show(500, truncate=False)

#---------------------------------------
print('State 5')
calculate_dispersion_type = types.ArrayType(types.StructType([types.StructField('neighbor_id', types.StringType()), types.StructField('dispersion', types.IntegerType())]))
def calculate_dispersion(same_neighbors):
  retL = []
  same_neighbors_dict = {}
  for item in same_neighbors:
    same_neighbors_dict[item['neighbor_id']] = set(item['neighbor_neighbors'])
  
  for same_neighbor in same_neighbors:
    dispersion = 0
    dic = {'neighbor_id': same_neighbor['neighbor_id']}
    comb = combinations(same_neighbor['neighbor_neighbors'], 2)
    for s,t in comb:
      if s in same_neighbors_dict[t]:
        continue
      if len(same_neighbors_dict[s].intersection(same_neighbors_dict[t])) > 2:
        continue
      dispersion += 1

    dic['dispersion'] = dispersion
    retL.append(dic)

  return retL

calculate_dispersion_udf = F.udf(calculate_dispersion, calculate_dispersion_type)

vertices = vertices.withColumn('dispersions', calculate_dispersion_udf(vertices['same_neighbors'])).drop('same_neighbors')
vertices.show(500, truncate=False)

#------------------------------------------
print('State 6')
max_dispersion_type = types.StructType([types.StructField('neighbor_ids', types.ArrayType(types.StringType())), types.StructField('dispersion', types.IntegerType())])
def max_dispersion(dispersions):

  max_disp = -1
  neighbors = None
  for item in dispersions:
    neighbor_id = item['neighbor_id']
    disp = item['dispersion']
    if disp > max_disp:
      max_disp = disp
      neighbors = [neighbor_id]
    elif disp == max_disp:
      neighbors.append(neighbor_id)

  return {'neighbor_ids': neighbors, 'dispersion': max_disp}

max_dispersion_udf = F.udf(max_dispersion, max_dispersion_type)

vertices = vertices.withColumn('max_dispersion', max_dispersion_udf(vertices['dispersions'])).drop('dispersions')
vertices.show(500, truncate=False)
