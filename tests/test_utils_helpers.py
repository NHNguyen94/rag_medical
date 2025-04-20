from src.utils.helpers import load_yml_configs


class TestUtilsHelper:
    test_yml_path = "tests/resources/test.yml"

    def test_load_configs(self):
        configs = load_yml_configs(self.test_yml_path)
        assert configs["test"] == "this is test yml file"
