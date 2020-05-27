"""
This is a top-level module
"""

class SampleClass:
    def __init__(self):
        """
        sample docstring
        """
        self.hello = "hello "

    def sample_method(self, name: str):
        """
        >>> from kale.sample_module import SampleClass
        >>>
        >>> greeter = SampleClass()
        >>> greeter.sample_method("Miguel and Mischa")
        'hello Miguel and Mischa'

        :param name:
        :return:
        """
        return self.hello + name
