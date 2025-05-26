from src.core_managers.chat_manager import ChatManager
import sys

def print_topic_info(topic_id, topic_name, similar_questions):
    """Print formatted topic information."""
    print("\n" + "="*50)
    print(f"Topic ID: {topic_id}")
    print(f"Topic Name: {topic_name}")
    if similar_questions:
        print("\nSimilar Questions:")
        for i, question in enumerate(similar_questions, 1):
            print(f"{i}. {question}")
    print("="*50 + "\n")

def print_topic_statistics(stats):
    """Print formatted topic statistics."""
    print("\n" + "="*50)
    print("Topic Statistics")
    print("="*50)
    print(f"Total Topics: {stats['total_topics']}")
    print("\nTop 10 Topics by Question Count:")
    sorted_topics = sorted(
        [(topic_id, count, stats['topic_names'][topic_id]) 
         for topic_id, count in stats['topic_distribution'].items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for topic_id, count, name in sorted_topics:
        print(f"Topic {topic_id} ({name}): {count} questions")
    print("="*50 + "\n")

def main():
    """Main CLI interface."""
    chat_manager = ChatManager()
    
    print("Medical Question Topic Analyzer")
    print("="*50)
    print("Commands:")
    print("  ask <question>  - Ask a medical question")
    print("  stats          - Show topic statistics")
    print("  exit           - Exit the program")
    print("="*50)
    
    while True:
        try:
            command = input("\nEnter command: ").strip()
            
            if command.lower() == 'exit':
                print("Goodbye!")
                break
                
            elif command.lower() == 'stats':
                stats = chat_manager.get_topic_statistics()
                if "error" in stats:
                    print(f"Error: {stats['error']}")
                else:
                    print_topic_statistics(stats)
                    
            elif command.lower().startswith('ask '):
                question = command[4:].strip()
                if not question:
                    print("Please provide a question after 'ask'")
                    continue
                    
                result = chat_manager.process_user_message(question)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print_topic_info(
                        result['topic_id'],
                        result['topic_name'],
                        result['similar_questions']
                    )
                    
            else:
                print("Unknown command. Available commands: ask, stats, exit")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 