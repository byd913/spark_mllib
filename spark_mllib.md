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
## local matrix
## distributed matrix


# Basic Statistics

# Classification and Regression


# Collaborative Filtering

# Clustering

# Dimensitionality Reducuction

# Feature Extraction and Tranformation

# Frequent Pattern Mining

# Evaluation Matrics

# PMML model export

# Optimization