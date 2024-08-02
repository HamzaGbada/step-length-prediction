Sure, I can provide Python implementations for cross-correlation, Pearson correlation coefficient, and normalized cross-correlation using commonly available libraries such as NumPy and SciPy.

### 1. Cross-Correlation

Here’s how you can compute the cross-correlation of two signals using SciPy:

```python
import numpy as np
from scipy.signal import correlate

def compute_cross_correlation(x, y):
    # Compute cross-correlation
    corr = correlate(x, y, mode='full')
    lag = np.arange(-(len(x) - 1), len(y))  # Lag values
    return lag, corr

# Example usage
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4])
lag, corr = compute_cross_correlation(x, y)

print("Lags:", lag)
print("Cross-correlation:", corr)
```

### 2. Pearson Correlation Coefficient

You can compute the Pearson correlation coefficient using NumPy:

```python
import numpy as np

def compute_pearson_correlation(x, y):
    # Compute Pearson correlation coefficient
    return np.corrcoef(x, y)[0, 1]

# Example usage
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])
pearson_corr = compute_pearson_correlation(x, y)

print("Pearson correlation coefficient:", pearson_corr)
```

### 3. Normalized Cross-Correlation

For normalized cross-correlation, you can normalize the cross-correlation values by dividing by the product of the norms of the signals:

```python
import numpy as np
from scipy.signal import correlate

def compute_normalized_cross_correlation(x, y):
    # Compute cross-correlation
    corr = correlate(x, y, mode='full')
    norm_x = np.sqrt(np.sum(x**2))
    norm_y = np.sqrt(np.sum(y**2))
    norm_corr = corr / (norm_x * norm_y)
    lag = np.arange(-(len(x) - 1), len(y))  # Lag values
    return lag, norm_corr

# Example usage
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4])
lag, norm_corr = compute_normalized_cross_correlation(x, y)

print("Lags:", lag)
print("Normalized cross-correlation:", norm_corr)
```

### Summary

1. **Cross-Correlation**: Measures how one signal correlates with another over different time lags.
2. **Pearson Correlation Coefficient**: Measures the linear relationship between two signals.
3. **Normalized Cross-Correlation**: Similar to cross-correlation but normalized to remove the effect of signal magnitude.

You can run these functions with your own signal data to analyze the relationships between them.