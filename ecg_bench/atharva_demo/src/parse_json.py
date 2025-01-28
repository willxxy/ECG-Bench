# Imports
import json

# VARIABLES/CONSTANTS
# JSON_PATH = "/mnt/newhome/atharva/projects/ECG-Byte/ecg_byte/data/ecg_chat_data/pretrain_mimic.json"
JSON_PATH = "./data/ecg_instruct_45k.json"

# ECG signal
def parse_ecg_signal(ecg_file):
    ecg_signal = "[BOS]Signal: " + ecg_file + "[EOS]"
    return ecg_signal


# Read JSON
def parse_conversation_dict_to_string(cur_conversation):
    # print("Length of current conversation:",len(cur_conversation["conversations"]))
    agents = {"h":"Human", "g":"GPT"}

    conversation_string = ""
    for message in cur_conversation["conversations"]:
        conversation_string += agents.get(message["from"][0],"Unknown") + ": " + message["value"] + "\n"

    ecg_signal = parse_ecg_signal(cur_conversation["ecg"])
    conversation_string = conversation_string.replace("<ecg>",ecg_signal)

    return conversation_string

def parse_conversation_dict_to_list(cur_conversation):
    # print("Length of current conversation:",len(cur_conversation["conversations"]))
    agents = {"h":"Human", "g":"GPT"}

    conversation_list = []
    for message in cur_conversation["conversations"]:
        conversation_list.append("[BOM]" + agents.get(message["from"][0],"Unknown") + ": " + message["value"] + "[EOM]") 

    # Add ECG Signal to return object
    ecg_signal = parse_ecg_signal(cur_conversation["ecg"])
    conversation_list[0] = conversation_list[0].replace("<ecg>",ecg_signal)

    # for idx,ele in enumerate(conversation_list):
    #     print(idx+1,ele)
    return conversation_list



def parse_json(file_path = JSON_PATH):
    """
    Reads a JSON file and returns its content as a list.
    
    :param file_path: Path to the JSON file.
    :return: List containing the JSON data.
    :raises ValueError: If the JSON content is not a list.
    :raises FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print("Example:",data[0],end = "\n")
            print("\n")
            print("="*100)
            # print("\n")
            
            # conversations = [x["conversations"] for x in data[0:1000] if ]
            conversations = [x for x in data if len(x["conversations"]) > 25]
            # conversations = data[:5]

            # print(conversations)
            for conv in conversations:
                print(conv["conversations"])
                print("\n")
                # print(parse_conversation_dict_to_string(conv))
                print(parse_conversation_dict_to_list(conv))
                print("-"*100)
                break
            
            quit()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        raise


if __name__ == "__main__":
    my_list = parse_json(JSON_PATH)
    print(my_list)
