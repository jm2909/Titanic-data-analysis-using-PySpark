import pandas as pd
import numpy as np
import csv
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
from pyspark.sql import Row
import pyspark.sql.functions as psf
from pyspark.sql.functions import *
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import UserDefinedFunction,struct
from pyspark.sql.types import IntegerType,StringType
from pyspark.mllib.evaluation import BinaryClassificationMetrics,MulticlassMetrics
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer,VectorAssembler,OneHotEncoder,QuantileDiscretizer,Imputer
from pyspark.sql.types import DoubleType

Tag = ['A5','PC','STON','CA','WEP','S0C','SOTON','WC','SCParis','A4','FCC','SCAH','SCPARIS','WEP','S0','PP']
CabinTag = ['A','B','C','D','E','F','No']
Nametag = [ 'Capt.','Dr.','Master.','Miss.','Mr.','Mrs.','Ms.','Rev']
EmbarkTag = ['C','Q','S']

df = spark.read.format("csv").option("header", "true").load('train.csv')
newdf = df.withColumn('FamilyMember', df['SibSp']+ df['Parch'])

def tagticket(row):
    c = ''
    for k in Tag:
        if str(row).replace("/",'').replace('.','').__contains__(k):
            c = k
            break
    if c !='':
        return c
    else:
        return ['Other']

def tagCabin(row):
    c = ''
    for k in CabinTag:
        if row !='':
            if row.__contains__(k):
                c = k
                break
        if c !='':
            return c
    else:
        return 'Other'


def tagName(row):
    c = ''
    for k in Nametag:
        if row != '':
            if row.Name.split(" ").__contains__(k):
                c = k
                break
    if c !='':
        return c
    else:
        return 'Other'
def scorecutoff(x):
    if x >= 0.40:
        return 1
    else:
        return 0

udfticket = UserDefinedFunction(tagticket, StringType())
udfcabin = UserDefinedFunction(tagCabin, StringType())
udfname = UserDefinedFunction(tagName, StringType())
newdf = newdf.withColumn('Nametag', udfname(struct(newdf['Name'])))
newdf = newdf.withColumn('Tickettag', udfticket(struct(newdf['Ticket'])))
newdf = newdf.withColumn('Cabintag', udfcabin(struct(newdf['Cabin'])))

stringIndexer0 = StringIndexer(inputCol="Sex", outputCol="SexCategory")
model0 = stringIndexer0.fit(newdf)
stringIndexer1 = StringIndexer(inputCol="Tickettag", outputCol="categoryIndexTicket")
model1 = stringIndexer1.fit(newdf)
stringIndexer2 = StringIndexer(inputCol="Cabintag", outputCol="categoryIndexCabin")
model2 = stringIndexer2.fit(newdf)
stringIndexer3 = StringIndexer(inputCol="Nametag", outputCol="categoryIndexName")
model3 = stringIndexer3.fit(newdf)
indexed = model0.transform(newdf)
indexed = model1.transform(indexed)
indexed = model2.transform(indexed)
newdf = model3.transform(indexed)

newdf = newdf.withColumn("Age", newdf["Age"].cast(DoubleType()))
imputer = Imputer(inputCols=["Age"], outputCols=["out_Age"], strategy='median')
newdf = imputer.fit(newdf).transform(newdf)
QuantileDiscreete1 = QuantileDiscretizer(numBuckets=5, inputCol="out_Age", outputCol="agebucket")
newdf = QuantileDiscreete1.fit(newdf).transform(newdf)

stringIndexer4 = StringIndexer(inputCol="out_Age", outputCol="categoryAge")
model4 = stringIndexer4.fit(newdf)
newdf = model4.transform(newdf)

newdf = newdf.withColumn("Fare", newdf["Fare"].cast(DoubleType()))
imputer2 = Imputer(inputCols=["Fare"], outputCols=["out_Fare"], strategy='median')
newdf = imputer2.fit(newdf).transform(newdf)
QuantileDiscreete2 = QuantileDiscretizer(numBuckets=4, inputCol="out_Fare", outputCol="Farebucket")
newdf = QuantileDiscreete2.fit(newdf).transform(newdf)

stringIndexer5 = StringIndexer(inputCol="out_Fare", outputCol="categoryFare")
model5 = stringIndexer5.fit(newdf)
newdf = model5.transform(newdf)

encoder = OneHotEncoder(inputCol="categoryFare", outputCol="categoryFarevec")
encoded = encoder.transform(newdf)
encoder = OneHotEncoder(inputCol="categoryAge", outputCol="categoryAgeVec")
encoded = encoder.transform(encoded)
encoder = OneHotEncoder(inputCol="categoryIndexName", outputCol="categoryVecName")
encoded = encoder.transform(encoded)
encoder = OneHotEncoder(inputCol="categoryIndexTicket", outputCol="categoryticket")
encoded = encoder.transform(encoded)
encoder = OneHotEncoder(inputCol="SexCategory", outputCol="SexCategoryvec")
encoded = encoder.transform(encoded)
# Emb and cabin need to add
cols_now = ['Survived', 'SexCategoryvec', 'categoryticket', 'categoryAgeVec', 'categoryFarevec', 'categoryVecName',
'FamilyMember']
encoded = encoded.select([c for c in encoded.columns if c in cols_now])
assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
encoded = encoded.withColumn("Survived", encoded["Survived"].cast(DoubleType()))
labelIndexer = StringIndexer(inputCol='Survived', outputCol="label")
tmp = [assembler_features, labelIndexer]
pipeline = Pipeline(stages=tmp)

allData = pipeline.fit(encoded).transform(encoded)

trainingData, testData = allData.randomSplit([0.8, 0.2], seed=0)  # need to ensure same split for each time

rf = RF(labelCol='label', featuresCol='features', numTrees=200)
fit = rf.fit(trainingData)
transformed = fit.transform(testData)
results = transformed.select(['probability', 'label'])
results_collect = results.collect()
results_list = [(float(i[0][0]), 1.0 - float(i[1])) for i in results_collect]
scoreAndLabels = sc.parallelize(results_list)

metrics = BinaryClassificationMetrics(scoreAndLabels)

pred_collect = transformed.select(['label', 'prediction', 'probability']).collect()
ytrue = [i[0] for i in pred_collect]
ypred = [i[1] for i in pred_collect]
probs = [scorecutoff(i[2][1]) for i in pred_collect]
print("The areaUnderPR score is (@numTrees=200)::::::::::: ", metrics.areaUnderPR)
print("The areaUnderROC score is (@numTrees=200)::::::::::::::: ", metrics.areaUnderROC)
print "With usual transformation:::::::::::::::", pd.crosstab(np.array(ytrue), np.array(ypred))
print "With cutoff transformation::::::::::::::", pd.crosstab(np.array(ytrue), np.array(probs))
fit.save('Titanic_Model')
print  " Model saved in the working directory"



