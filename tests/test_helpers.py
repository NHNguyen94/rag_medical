from src.utils.helpers import check_if_dir_exists


class TestUtilsHelper:
    path_exists = "tests/resources"
    path_not_exists = "tests/resources/does_not_exist"

    def test_check_if_dir_exists(self):
        assert check_if_dir_exists(self.path_exists) is True

    def test_check_if_dir_not_exists(self):
        assert check_if_dir_exists(self.path_not_exists) is False
