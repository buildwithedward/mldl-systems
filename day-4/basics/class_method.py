class Student:
    school = "ABC School"

    @classmethod
    def get_school_name(cls):
        """Class method to get the name of the school. 
        It uses the @classmethod decorator, which allows it to access class-level attributes like 'school'."""
        return cls.school
    
# Example usage
print(Student.get_school_name())  # Output: ABC School