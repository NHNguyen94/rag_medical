from src.utils.helpers import load_yml_configs, clean_text


class TestUtilsHelper:
    test_yml_path = "tests/resources/test.yml"

    def test_load_configs(self):
        configs = load_yml_configs(self.test_yml_path)
        assert configs["test"] == "this is test yml file"

    def test_clean_text(self):
        text = "  This is a test text. \n\n It has multiple lines and spaces, with numbers 123 and special characters !@#."
        cleaned_text = clean_text(text)
        print(cleaned_text)
        assert "123" not in cleaned_text, "Numbers should be removed"
        assert "!@#" not in cleaned_text, "Special characters should be removed"
