class Calculator:
    """Static methods for basic arithmetic operations. These methods can be called without creating an instance of the class."""
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def subtract(a, b):
        return a - b
    
# Using the static methods without creating an instance of the class
result_add = Calculator.add(5, 3)
result_subtract = Calculator.subtract(10, 4)
print(f"Addition Result: {result_add}")        # Output: Addition Result: 8
print(f"Subtraction Result: {result_subtract}")  # Output: Subtraction Result: 6