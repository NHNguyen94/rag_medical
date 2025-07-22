from src.utils.helpers import load_yml_configs, clean_text, write_dict_to_yaml


class TestUtilsHelper:
    test_yml_path = "tests/resources/test.yml"

    def test_load_configs(self):
        configs = load_yml_configs(self.test_yml_path)
        assert configs["test"] == "this is test yml file"

    def test_write_dict_to_yaml(self):
        test_data = {
            "test_field_1": "test_value_1",
            "test_field_2": "test_value_2",
            "test_field_3": "test_value_3",
        }
        write_dict_to_yaml(test_data, "tests/output/test_yml_output.yml")

    def test_clean_text(self):
        text = "  This is a test text. \n\n It has multiple lines and spaces, with numbers 123 and special characters !@#."
        cleaned_text = clean_text(text)
        print(cleaned_text)
        assert "123" not in cleaned_text, "Numbers should be removed"
        assert "!@#" not in cleaned_text, "Special characters should be removed"
