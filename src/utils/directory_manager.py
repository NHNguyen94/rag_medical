import os
from typing import Dict
from typing import List, Union

import pandas as pd


class DirectoryManager:
    @staticmethod
    def print_current_dir() -> None:
        print(os.getcwd())

    @staticmethod
    def check_if_dir_exists(dir: str) -> bool:
        return os.path.exists(dir)

    @staticmethod
    def create_dir_if_not_exists(dir: str) -> None:
        if not DirectoryManager.check_if_dir_exists(dir):
            os.makedirs(dir)
            print(f"Directory {dir} created")
        else:
            print(f"Directory {dir} already exists")

    @staticmethod
    def check_if_file_exists(file_path: str) -> bool:
        return os.path.exists(file_path)

    @staticmethod
    def check_if_empty_file(file_path: str) -> bool:
        return os.path.exists(file_path) and os.path.getsize(file_path) == 0

    @staticmethod
    def move_file(src: str, dest: str) -> None:
        os.rename(src, dest)
        print(f"File {src} moved to {dest}")

    @staticmethod
    def delete_file(file_path: str) -> None:
        os.remove(file_path)
        print(f"File {file_path} deleted")

    @staticmethod
    def delete_empty_dir(dir: str) -> None:
        os.rmdir(dir)
        print(f"Directory {dir} deleted")

    @staticmethod
    def get_file_path_in_dir(folder_path: str) -> Union[List[str], None]:
        if DirectoryManager.check_if_dir_exists(folder_path):
            return [
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, file))
            ]
        else:
            return None

    @staticmethod
    def get_all_recursive_files(folder_path: str, file_extension: str) -> List[str]:
        files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.endswith(file_extension):
                    files.append(os.path.join(root, filename))
        return files

    @staticmethod
    def delete_non_empty_dir(path: str) -> None:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(path)

    @staticmethod
    def create_empty_csv_file(col_names: List, file_path: str) -> None:
        df = pd.DataFrame(columns=col_names)
        df.to_csv(file_path, index=False)
        print("Empty CSV file created with columns:", col_names)

    @staticmethod
    def write_log_file(log_file_path: str, data: Dict) -> None:
        if not DirectoryManager.check_if_file_exists(log_file_path):
            col_names = list(data.keys())
            DirectoryManager.create_empty_csv_file(
                col_names=col_names, file_path=log_file_path
            )
        df = pd.DataFrame([data])
        df.to_csv(log_file_path, mode="a", index=False, header=False)
