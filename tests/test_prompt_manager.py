from src.core_managers.prompt_manager import PromptManager


class TestPromptManager:
    prompt_path = "tests/resources/test_prompt.yml"
    prompt_manager = PromptManager(prompt_path)

    def test_get_system_prompt(self):
        assert (
            self.prompt_manager.get_system_prompt() == "You are a medical assistant.\n"
        )

    def test_get_reasoning_effort(self):
        assert self.prompt_manager.get_reasoning_effort() == "low"

    def test_get_temperature(self):
        assert self.prompt_manager.get_temperature() == 0.7
