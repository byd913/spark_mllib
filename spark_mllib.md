# DataTypes
## local vector
### dense vector
* numpy array
```python
dv1 = np.array([1.0, 0.0, 3.0])
```
* python list
```python
dv2 = [1.0, 0.0, 3.0]
```

### sparse vectors
* MLlib SparseVector
```python
from pyspark.mllib.linalg import Vectors
sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])
```
* Scipy csc_matrix with  sign column
```python
sv2 = sps.csc_matrix((np.array([1.0, 3.0]), np.array([0, 2]), np.array([0, 2])), shape=(3, 1))
```

## labled point
```python
pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))
```

## local matrix
stored on a single machine
```python
from pyspark.mllib.linalg import Matrix, Matrices
dm2 = Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])
```

## distributed matrix
stored distributively in one or more RDDs. matrix must be deterministic.

### RowMatrix
row-orient matrix. each row is a local vector. without row index
```python
rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
mat = RowMatrix(rows)
```
### IndexedRowMatrix
similar to RowMatrix. With row index
```python
indexedRows = sc.parallelize([(0, [1, 2, 3]), (1, [4, 5, 6]),
                              (2, [7, 8, 9]), (3, [10, 11, 12])])
mat = IndexedRowMatrix(indexedRows)
```
### CoordinateMatrix
with entries, each entry is a tuple of (i: Long, j:Long, value: Double), i is row index, j is col index, value is entry value.
used when both dimensions of the matrx is huge and the matrix is sparse.
```python
from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry
entries = sc.parallelize([MatrixEntry(0, 0, 1.2), MatrixEntry(1, 0, 2.1), MatrixEntry(6, 1, 3.7)])
entries = sc.parallelize([(0, 0, 1.2), (1, 0, 2.1), (2, 1, 3.7)])
mat = CoordinateMatrix(entries)
```

### BlockMatrix
matrix backed by an RDD of MatrixBlocks. MatrixBlocks is a tuple of ((Int, Int), Matrix), (Int, Int) is the index of the block.
Matrix is the sub-matrix at the given index with size rowsPerBlock x colsPerBlock.
```python
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import BlockMatrix
blocks = sc.parallelize([((0, 0), Matrices.dense(3, 2, [1, 2, 3, 4, 5, 6])),
                         ((1, 0), Matrices.dense(3, 2, [7, 8, 9, 10, 11, 12]))])
mat = BlockMatrix(blocks, 3, 2)
m = mat.numRows()  # 6
n = mat.numCols()  # 2
```

# Basic Statistics
## Summary Statisticse
colStats() return an instance of MultivariateStatisticalSummary, contain the column-wise max, min, mean, variance and number of nonzeros
```python
from pyspark.mllib.stat import Statistics
mat = sc.parallelize(
    [np.array([1.0, 10.0, 100.0]), np.array([2.0, 20.0, 200.0]), np.array([3.0, 30.0, 300.0])]
)
summary = Statistics.colStats(mat)
print summary.mean() 
print summary.max()
```
### Correlations
calculate correction between two series. support Pearson and Spearman correction.
```python
from pyspark.mllib.stat import Statistics
seriesX = sc.parallelize([1.0, 2.0, 3.0, 3.0, 5.0]) 
seriesY = sc.parallelize([11.0, 22.0, 33.0, 33.0, 555.0])
print "Correlation is: " + str(Statistics.corr(seriesX, seriesY, method="pearson"))
```

### Random data generation

### Stratified sampling

### Hypothesis testing

### Streaming Significance Testing

### Kernel density estimation



# Classification and Regression
## Linear Model
### Mathematical formulation
#### Loss functions
* hinge loss
* logistic loss
* squared loss

#### Regularizers
* zero (unregularazed)
* L2
* L1
* elastic net

#### Optimization
* SGD (most)
* L-BFGS (a few)

### Classification
#### Linear Support Vector Machines
```python
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile("data/mllib/sample_svm_data.txt")
parsedData = data.map(parsePoint)

model = SVMWithSGD.train(parsedData, iterations=100)

labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))

model.save(sc, "target/tmp/pythonSVMWithSGDModel")
sameModel = SVMModel.load(sc, "target/tmp/pythonSVMWithSGDModel")
```
#### Logistic regression

### Regression
#### Linear least squares, Lasso and ridge regression
#### Stream linear regression

### Implementation

## Decision trees

## Ensembles of decision trees

## Naive Bayes

## Isotonic regression

# Collaborative Filtering

# Clustering

# Dimensitionality Reducuction

# Feature Extraction and Tranformation

# Frequent Pattern Mining

# Evaluation Matrics

# PMML model export

# Optimization
