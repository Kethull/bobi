# physics_utils.py

import numpy as np

class PhysicsError(Exception):
    """Custom exception for physics-related errors, including numerical issues."""
    pass

def safe_divide(numerator, denominator, epsilon=1e-12, default_on_zero_denom=0.0):
    """
    Safely divides two numbers, handling potential division by zero.

    Args:
        numerator (float or np.ndarray): The number(s) to be divided.
        denominator (float or np.ndarray): The number(s) to divide by.
        epsilon (float): Threshold below which the denominator is considered zero.
        default_on_zero_denom (float): Value to return if denominator is effectively zero.
                                       Can be set to float('inf') or float('-inf') based on numerator's sign
                                       if that behavior is desired, or a specific numeric value.

    Returns:
        float or np.ndarray: The result of the division, or default_on_zero_denom if denominator is near zero.
    """
    if isinstance(denominator, (np.ndarray)):
        is_zero = np.abs(denominator) < epsilon
        # Create an array of default values for zero denominator cases
        # Handle sign of numerator for inf/-inf defaults if desired
        if default_on_zero_denom == float('inf') or default_on_zero_denom == float('-inf'):
            default_vals = np.where(numerator > 0, float('inf'), 
                                   np.where(numerator < 0, float('-inf'), 0.0)) # 0/0 = 0
        else:
            default_vals = np.full_like(denominator, default_on_zero_denom, dtype=np.float64)

        # Perform division where denominator is not zero, otherwise use default_val
        # np.divide has an 'out' argument to store results and 'where' to control operation
        result = np.divide(numerator, denominator, out=np.zeros_like(denominator, dtype=np.float64), where=~is_zero)
        result[is_zero] = default_vals[is_zero] if isinstance(default_vals, np.ndarray) else default_vals
        return result
    else: # Scalar case
        if abs(denominator) < epsilon:
            if default_on_zero_denom == float('inf') or default_on_zero_denom == float('-inf'):
                if abs(numerator) < epsilon: # 0/0 case
                    return 0.0
                return float('inf') if numerator > 0 else float('-inf')
            return default_on_zero_denom
        return numerator / denominator

def normalize_vector(vector, epsilon=1e-12):
    """
    Normalizes a vector to unit length.

    Args:
        vector (np.ndarray): The vector to normalize.
        epsilon (float): Threshold below which the vector's magnitude is considered zero.

    Returns:
        np.ndarray: The normalized vector, or the original vector (or zero vector)
                    if its magnitude is close to zero.
    """
    if not isinstance(vector, np.ndarray):
        vector = np.array(vector, dtype=float)

    norm = np.linalg.norm(vector)
    if norm < epsilon:
        # Return a zero vector of the same shape if the norm is too small
        return np.zeros_like(vector)
    return vector / norm

if __name__ == '__main__':
    # Test cases for safe_divide
    print("--- Testing safe_divide ---")
    print(f"5 / 2 = {safe_divide(5, 2)}")
    print(f"5 / 0 (default 0.0) = {safe_divide(5, 0)}")
    print(f"5 / 1e-15 (default 0.0) = {safe_divide(5, 1e-15)}")
    print(f"0 / 0 (default 0.0) = {safe_divide(0, 0)}")
    print(f"5 / 0 (default inf) = {safe_divide(5, 0, default_on_zero_denom=float('inf'))}")
    print(f"-5 / 0 (default inf) = {safe_divide(-5, 0, default_on_zero_denom=float('inf'))}") # Will be -inf
    print(f"0 / 0 (default inf) = {safe_divide(0, 0, default_on_zero_denom=float('inf'))}") # Will be 0

    # Numpy array tests
    num_arr = np.array([1.0, 0.0, -5.0, 6.0])
    den_arr1 = np.array([2.0, 1e-15, 2.0, 0.0])
    print(f"Array division (default 0.0): {safe_divide(num_arr, den_arr1)}")
    print(f"Array division (default inf): {safe_divide(num_arr, den_arr1, default_on_zero_denom=float('inf'))}")

    den_arr2 = np.array([0.0, 0.0, 0.0, 0.0])
    print(f"Array division by all zeros (default 0.0): {safe_divide(num_arr, den_arr2)}")
    print(f"Array division by all zeros (default inf): {safe_divide(num_arr, den_arr2, default_on_zero_denom=float('inf'))}")

    # Test cases for normalize_vector
    print("\n--- Testing normalize_vector ---")
    print(f"Normalize [3, 4]: {normalize_vector(np.array([3.0, 4.0]))}")
    print(f"Normalize [0, 0]: {normalize_vector(np.array([0.0, 0.0]))}")
    print(f"Normalize [1e-15, 1e-15]: {normalize_vector(np.array([1e-15, 1e-15]))}")
    print(f"Normalize [10, 0, 0]: {normalize_vector(np.array([10.0, 0.0, 0.0]))}")
    vec_list = [1,1,1]
    print(f"Normalize list [1,1,1]: {normalize_vector(vec_list)}")