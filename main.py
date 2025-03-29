from .config import Config
from .chat_agent import CodeAssistAgent
from .chat_interface import ChatInterface

def main():
    config = Config()
    agent = CodeAssistAgent(config)
    interface = ChatInterface(agent)
    interface.start_chat()

if __name__ == "__main__":
    main()
