from utils import prepare_imports
prepare_imports()

from kale.sample_module import SampleClass
from config import get_config

if __name__ == "__main__":

    c = get_config()
    assert c.sample_key == "sample_value"
    print(SampleClass().sample_method("Miguel and Mischa"))
    print("Your library project kale is done waiting for you!")