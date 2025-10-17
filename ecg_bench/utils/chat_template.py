"""From https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
Modified by @willxxy
"""

import base64
import dataclasses
import os
from enum import IntEnum, auto
from io import BytesIO
from typing import Dict, List, Tuple, Union


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    LLAMA3 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    DEEPSEEK_CHAT = auto()
    METAMATH = auto()
    YUAN2 = auto()
    GEMMA = auto()
    CLLM = auto()
    DEFAULT = auto()


IMAGE_PLACEHOLDER_STR = "$$<image>$$"

LOGDIR = os.getenv("LOGDIR", ".")


class ImageFormat(IntEnum):
    """Image formats."""

    URL = auto()
    LOCAL_FILEPATH = auto()
    PIL_IMAGE = auto()
    BYTES = auto()
    DEFAULT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    system_message_vision: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    # Each message is either a string or a tuple of (string, List[image_url]).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None
    # The maximum image size in megabytes that this model takes in. None means we do not resize the image.
    max_image_size_mb: int = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        if self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        if self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        if self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        if self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        if self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = IMAGE_PLACEHOLDER_STR * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        if self.sep_style == SeparatorStyle.LLAMA3:
            ret = "<|begin_of_text|>"
            if self.system_message:
                ret += system_prompt
            else:
                ret += ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                    ret += f"{message.strip()}<|eot_id|>"
                else:
                    ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            return ret
        if self.sep_style == SeparatorStyle.DEEPSEEK_CHAT:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        if self.sep_style == SeparatorStyle.GEMMA:
            ret = "<bos>"
            for role, message in self.messages:
                if message:
                    ret += "<start_of_turn>" + role + "\n" + message + self.sep
                else:
                    ret += "<start_of_turn>" + role + "\n"
            # For gemma-2, append <eos> once at the very end
            if self.stop_str == "<eos>":
                ret += "<eos>"
            return ret
        if self.sep_style == SeparatorStyle.DEFAULT:
            ret = system_prompt + "\n"
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                    ret += role + ": " + message + "\n"
                else:
                    ret += role + ":"
            return ret
        raise ValueError(f"Invalid style: {self.sep_style}")

    def get_images(self):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    for image in msg[1]:
                        images.append(image.base64_str)

        return images

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def get_system_message(self, is_vision=False):
        """Return the system message."""
        if is_vision and self.system_message_vision:
            return self.system_message_vision
        return self.system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def save_new_images(self, has_csam_images=False):
        import hashlib

        _, last_user_message = self.messages[-2]

        if type(last_user_message) is tuple:
            _, images = last_user_message[0], last_user_message[1]

            image_directory_name = "csam_images" if has_csam_images else "serve_images"
            for image in images:
                loaded_image = self.load_image(image.base64_str)
                hash_str = hashlib.md5(loaded_image.tobytes()).hexdigest()
                filename = os.path.join(
                    image_directory_name,
                    f"{hash_str}.{image.filetype}",
                )

                filename = os.path.join(LOGDIR, filename)
                if not os.path.isfile(filename):
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    loaded_image.save(filename)

    def extract_text_and_image_hashes_from_messages(self):
        import hashlib

        messages = []

        for role, message in self.messages:
            if type(message) is tuple:
                text, images = message[0], message[1]

                image_hashes = []
                for image in images:
                    if image.image_format == ImageFormat.URL:
                        image_hashes.append(image)
                    elif image.image_format == ImageFormat.BYTES:
                        # Using the class's own load_image method instead of FastChat's
                        loaded_image = self.load_image(image.base64_str)
                        image_hash = hashlib.md5(loaded_image.tobytes()).hexdigest()
                        image_hashes.append(image_hash)

                messages.append((role, (text, image_hashes)))
            else:
                messages.append((role, message))

        return messages

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            system_message_vision=self.system_message_vision,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            max_image_size_mb=self.max_image_size_mb,
        )

    def load_image(self, image_file):
        import requests
        from PIL import Image

        image = None

        if image_file.startswith("http://") or image_file.startswith("https://"):
            timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
            response = requests.get(image_file, timeout=timeout)
            image = Image.open(BytesIO(response.content))
        elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
            image = Image.open(image_file)
        elif image_file.startswith("data:"):
            image_file = image_file.split(",")[1]
            image = Image.open(BytesIO(base64.b64decode(image_file)))
        else:
            image = Image.open(BytesIO(base64.b64decode(image_file)))

        return image

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.extract_text_and_image_hashes_from_messages(),
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# An empty template for raw conversation.
register_conv_template(
    Conversation(
        name="raw",
        system_message="",
        roles=("", ""),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
    ),
)

register_conv_template(
    Conversation(
        name="meta-llama-3.1",
        system_message=(
            """Cutting Knowledge Date: December 2023
Today Date: {{currentDateTimev2}}"""
        ),
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
    ),
)

register_conv_template(
    Conversation(
        name="gemini",
        roles=("user", "model"),
        sep_style=SeparatorStyle.DEFAULT,
        sep=None,
        max_image_size_mb=20,
    ),
)

# Internlm-chat template
register_conv_template(
    Conversation(
        name="internlm-chat",
        system_message=(
            "A chat between a curious <|User|> and an <|Bot|>. "
            "The <|Bot|> gives helpful, detailed, and polite answers "
            "to the <|User|>'s questions.\n\n"
        ),
        roles=("<|User|>", "<|Bot|>"),
        sep_style=SeparatorStyle.CHATINTERN,
        sep="<eoh>",
        sep2="<eoa>",
        stop_token_ids=[1, 103028],
        stop_str="<|User|>",
    ),
)

# Mistral template
# source: https://docs.mistral.ai/llm/mistral-instruct-v0.1#chat-template
register_conv_template(
    Conversation(
        name="mistral",
        system_template="[INST] {system_message}\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2="</s>",
    ),
)

# llama2 template
# reference: https://huggingface.co/blog/codellama#conversational-instructions
# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
register_conv_template(
    Conversation(
        name="llama-2",
        system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
        roles=("[INST]", "[/INST]"),
        sep_style=SeparatorStyle.LLAMA2,
        sep=" ",
        sep2=" </s><s>",
    ),
)

# llama3 template
# reference: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
# reference: https://github.com/meta-llama/llama3/blob/0cee08ec68f4cfc0c89fe4a9366d82679aaa2a66/llama/tokenizer.py#L222
register_conv_template(
    Conversation(
        name="llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep_style=SeparatorStyle.LLAMA3,
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    ),
)

# Qwen-chat default template
# source: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py#L130
register_conv_template(
    Conversation(
        name="qwen-7b-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    ),
)

# Deepseek-chat template
# reference: https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat/blob/main/tokenizer_config.json
register_conv_template(
    Conversation(
        name="deepseek-chat",
        system_message="<｜begin▁of▁sentence｜>",  # must add a bos token before first message
        roles=("User", "Assistant"),
        sep_style=SeparatorStyle.DEEPSEEK_CHAT,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_str="<｜end▁of▁sentence｜>",
    ),
)

# Gemma
# reference: https://huggingface.co/google/gemma-7b-it?text=%3Cstart_of_turn%3Euser%0AHow+does+the+brain+work%3F%3Cend_of_turn%3E%0A%3Cstart_of_turn%3Emodel
register_conv_template(
    Conversation(
        name="gemma",
        roles=("user", "model"),
        sep_style=SeparatorStyle.GEMMA,
        sep="<end_of_turn>\n",
        stop_str="<end_of_turn>",
    ),
)

# Gemma 2
# reference: https://github.com/huggingface/transformers/issues/32110
register_conv_template(
    Conversation(
        name="gemma-2",
        roles=("user", "model"),
        sep_style=SeparatorStyle.GEMMA,
        sep="<end_of_turn>\n",
        stop_str="<eos>",
    ),
)
