from src.utils.directory_manager import DirectoryManager


class TestDirectoryManager:
    non_exist_test_file_path = "tests/resources/test_folder_1/test_file_2.txt"
    exist_test_file_path = "tests/resources/test_folder_1/test_file_1.txt"
    non_exist_test_folder_path = "tests/resources/test_folder_2"
    exist_test_folder_path = "tests/resources/test_folder_1"
    empty_folder_path = "tests/resources/empty_folder"
    DirectoryManager.create_dir_if_not_exists(empty_folder_path)

    def test_check_if_dir_exists(self):
        assert DirectoryManager.check_if_dir_exists(self.exist_test_folder_path) == True
        assert (
            DirectoryManager.check_if_dir_exists(self.non_exist_test_folder_path)
            == False
        )

    def test_create_dir_if_not_exists(self):
        DirectoryManager.create_dir_if_not_exists(self.non_exist_test_folder_path)
        assert (
            DirectoryManager.check_if_dir_exists(self.non_exist_test_folder_path)
            == True
        )
        DirectoryManager.delete_empty_dir(self.non_exist_test_folder_path)
        assert (
            DirectoryManager.check_if_dir_exists(self.non_exist_test_folder_path)
            == False
        )

    def test_get_file_path_in_dir(self):
        assert DirectoryManager.get_file_path_in_dir(self.exist_test_folder_path) == [
            "tests/resources/test_folder_1/test_file_1.txt",
            "tests/resources/test_folder_1/test_file_2.txt",
            "tests/resources/test_folder_1/test_file_3.txt",
        ]
        assert DirectoryManager.get_file_path_in_dir(self.empty_folder_path) == []
        assert (
            DirectoryManager.get_file_path_in_dir(self.non_exist_test_folder_path)
            == None
        )
