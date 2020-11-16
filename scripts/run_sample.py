from kyle.sample_module import SampleClass
from utils import prepare_imports

from config import get_config

prepare_imports()


if __name__ == "__main__":

    c = get_config()
    assert c.sample_key == "sample_value"
    print(SampleClass().sample_method("Miguel and Mischa"))
    print("Your library project kyle is done waiting for you!")
