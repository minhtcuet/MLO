from numpy cimport ndarray as ar
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def convert2numpyarr(xy):
    cdef int i, j, h=len(xy), w=len(xy[0])
    cdef ar[object, ndim=2] new = np.empty((h, w), dtype=object)
    for i in xrange(h):
        for j in xrange(w):
            new[i,j] = xy[i][j]
    return new


cimport cython
import numpy as np
cimport numpy as np

# Import the necessary CatBoost classes
from catboost import CatBoostClassifier

# Define the Cython wrapper function
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef predict_proba_catboost( X, model):
    # Cast the model object to the appropriate type
    cdef void *model_ptr = <void *>model

    # Perform the prediction using the CatBoost predict_proba function
    cdef np.ndarray[np.float64_t, ndim=2] y_pred = model.predict_proba(X)


    return y_pred[:, 1].tolist()




# Define the Cython wrapper function
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef predict_catboost( X, model):
    # Cast the model object to the appropriate type
    cdef void *model_ptr = <void *>model

    # Perform the prediction using the CatBoost predict_proba function
    cdef np.ndarray y_pred = model.predict(X)

    return y_pred.flatten().tolist()