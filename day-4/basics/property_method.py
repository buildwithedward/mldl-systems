class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        """This is the getter method for the celsius property. 
        It allows us to access the value of _celsius as if it were a regular attribute."""
        return self._celsius 

    @celsius.setter
    def celsius(self, value):
        """This is the setter method for the celsius property. 
        It allows us to set the value of _celsius while also performing validation."""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero.")
        self._celsius = value

t = Temperature(30)
print(f"Using Clean getter: {t.celsius}°C")  # Output: 30°C
t.celsius = 25
print(f"Using Clean setter: {t.celsius}°C")  # Output: 25°C
t.celsius = -300  # This will raise a ValueError: Temperature cannot be below absolute zero.
print(f"Using Clean setter: {t.celsius}°C")  # This line will not be executed due to the exception.
