from kale.sample_module import SampleClass


def test_SampleClass():
    greeter = SampleClass()
    assert greeter.hello == "hello "