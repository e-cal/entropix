from pathlib import Path
from entropix.llama import LLAMA_1B_PARAMS
from entropix.model import load_weights, generate
from entropix.tokenizer import Tokenizer

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
print(prompt)

tokenizer = Tokenizer("entropix/tokenizer.model")
model = load_weights(Path("./weights/1B-Instruct"), LLAMA_1B_PARAMS.n_layers)
generate(model, LLAMA_1B_PARAMS, tokenizer, prompt, max_tokens=1024)
