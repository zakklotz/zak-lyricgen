from .config import Config
from .chat_agent import CodeAssistAgent
from .chat_interface import ChatInterface
from .generator import ResponseGenerator  # Optional: expose this
from .model_loader import load_components  # Optional: expose this


def run():
    config = Config()
    agent = CodeAssistAgent(config)
    interface = ChatInterface(agent)
    interface.start_chat()


if __name__ == "__main__":
    run()
