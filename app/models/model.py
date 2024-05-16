from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
import os
from app import PROJECT_PATH

def get_embed_model(model_path="local:BAAI/bge-m3"):
    embed_model = resolve_embed_model(model_path)
    return embed_model


def get_llm(model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            temperature=0.1,
            max_new_tokens=512,
            context_window=8000,
            n_gpu_layers=35):
    llm = LlamaCPP(
        model_path=model_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
        generate_kwargs={},
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": n_gpu_layers,"n_threads": int(os.cpu_count())},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True
    )
    return llm

embed_model = get_embed_model(model_path="local:BAAI/bge-m3")
llm = get_llm(model_path=os.path.join(PROJECT_PATH,"models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"), max_new_tokens=4096)