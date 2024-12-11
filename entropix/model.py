import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, NamedTuple, Optional, Tuple
import copy
import re
import jax.numpy as jnp
import numpy as np
from scipy.special import kv
import torch
import torch.nn.functional as F
from rich import print as rprint
from openai import OpenAI
from entropix.config import DEFAULT_MASK_VALUE, SamplerConfig, SamplerState, STATE_COLOR_MAP
from entropix.kvcache import KVCache
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer, Message
from entropix.metrics import AttnMetrics, TokenMetrics, calculate_metrics

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################################################################################
#                                    Types                                     #
################################################################################

class LayerWeights(NamedTuple):
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    ffn_norm: torch.Tensor
    attention_norm: torch.Tensor

class XfmrWeights(NamedTuple):
    tok_embeddings: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor
    layer_weights: list[LayerWeights]

class ModelParams(NamedTuple):
    name: str
    dim: int
    n_layers: int
    n_local_heads: int
    n_local_kv_heads: int
    head_dim: int
    max_seq_len: int
    rope_theta: float
    use_scaled_rope: bool
    hf_id: str | None = None

class Model(NamedTuple):
    weights: XfmrWeights
    params: ModelParams
    tokenizer: Tokenizer

@dataclass
class GenerationData:
    prompt: str
    response: str
    tokens: list[str]
    messages: list[Message]
    metrics: list[TokenMetrics]
    sampler_cfg: SamplerConfig
    sampler_states: list[SamplerState]

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "tokens": self.tokens,
            "messages": [m.model_dump() for m in self.messages],
            "metrics": [asdict(m) for m in self.metrics],
            # "sampler_cfg": asdict(self.sampler_cfg),
            "sampler_cfg": self.sampler_cfg.model_dump(),
            "sampler_states": [s.name for s in self.sampler_states],
        }

    def save(self, fp: str):
        with open(fp, "w") as f:
            s = json.dumps(self.to_dict())
            f.write(s)

    @classmethod
    def load(cls, fp: str):
        with open(fp, 'rb') as f:
            data = json.load(f)
        data["metrics"] = [TokenMetrics(**m) for m in data["metrics"]]
        data["messages"] = [Message(**m) for m in data["messages"]]
        data["sampler_cfg"] = SamplerConfig(**data["sampler_cfg"])
        data["sampler_states"] = [SamplerState[name] for name in data["sampler_states"]]
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        data["metrics"] = [TokenMetrics(**m) for m in data["metrics"]]
        data["messages"] = [Message(**m) for m in data["messages"]]
        # data["sampler_cfg"] = SamplerConfig(**data["sampler_cfg"])
        data["sampler_cfg"] = SamplerConfig.from_dict(data["sampler_cfg"])
        data["sampler_states"] = [SamplerState[name] for name in data["sampler_states"]]
        return cls(**data)

################################################################################
#                                   Weights                                    #
################################################################################
def load_weights(ckpt_dir: Path | str, model_cfg: ModelParams) -> XfmrWeights:
    print(f"Loading weights from {ckpt_dir}...")
    if isinstance(ckpt_dir, str): ckpt_dir = Path(ckpt_dir)
    w = {}
    layer_weights = []
    sep = '\\' if os.name == 'nt' else '/'
    with torch.inference_mode():
        for file in ckpt_dir.glob("*.npy"):
            name = '.'.join(str(file).split(sep)[-1].split('.')[:-1])
            jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
            np_weight = np.array(jax_weight).astype(np.float32)
            weight = torch.from_numpy(np_weight).to(torch.bfloat16).to(device)
            w[name] = weight.to(device)
        for i in range(model_cfg.n_layers):
            layer_weights.append(
                LayerWeights(
                    wq=w[f'layers.{i}.attention.wq.weight'],
                    wk=w[f'layers.{i}.attention.wk.weight'],
                    wv=w[f'layers.{i}.attention.wv.weight'],
                    wo=w[f'layers.{i}.attention.wo.weight'],
                    w1=w[f'layers.{i}.feed_forward.w1.weight'],
                    w2=w[f'layers.{i}.feed_forward.w2.weight'],
                    w3=w[f'layers.{i}.feed_forward.w3.weight'],
                    ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
                    attention_norm=w[f'layers.{i}.attention_norm.weight'],
                )
            )
        xfmr_weights = XfmrWeights(tok_embeddings=w['tok_embeddings.weight'], norm=w['norm.weight'], output=w['output.weight'], layer_weights=layer_weights)
        return xfmr_weights

################################################################################
#                                 Attention                                    #
################################################################################
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.to(dtype), xk_out.to(dtype)

def attention(
    x: torch.Tensor,
    layer_weights: LayerWeights,
    model_params,
    cur_pos: int,
    layer_idx: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    if x.dim() == 2:  # add batch dimension to 2d input
        bs = 1
        seq_len, dim = x.shape
        x = x.unsqueeze(0)
    else:
        bs, seq_len, dim = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = F.linear(x, layer_weights.wq).view(bs, seq_len, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, layer_weights.wk).view(bs, seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, layer_weights.wv).view(bs, seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=xq.dtype)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = xq.permute(0, 2, 1, 3)  # (bs, n_heads, seqlen, head_dim)
    keys = keys.permute(0, 2, 3, 1)  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = values.permute(0, 2, 1, 3)  # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.to(torch.float32)  # Always do attention softmax at float32
    if cur_pos == 0: scores = scores + attn_mask
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = F.softmax(padded_logits, dim=-1).to(x.dtype)
    output = torch.matmul(scores.to(values.dtype), values)
    output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
    out = F.linear(output, layer_weights.wo)
    # If input was 2D, remove the batch dimension from the output
    if dim == 2: out = out.squeeze(0)
    return out, kvcache, pre_scores

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3), layer_weights.w2)

################################################################################
#                                 Transformer                                  #
################################################################################

def xfmr(
    xfmr_weights: XfmrWeights,
    model_params: ModelParams,
    tokens: torch.Tensor,
    cur_pos: int,
    freqs_cis: torch.Tensor,
    kvcache: KVCache,
    attn_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnMetrics]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnMetrics.new(bsz=tokens.shape[0], n_layers=model_params.n_layers, n_heads=model_params.n_local_heads, device=device)
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:, :, -1, :], i)
        h = h + h_attn
        h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    return logits, kvcache, scores, attn_stats

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)

    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2, dtype=dtype, device=device)[:(dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
    else:
        raise ValueError("seqlen <= 1")
    return mask

@dataclass
class Branch:
    tokens: torch.Tensor | list
    kvcache: KVCache
    cur_pos: int
    tokens_text: list[str] = field(default_factory=list)
    metrics: list[TokenMetrics] = field(default_factory=list)
    sampler_states: list[SamplerState] = field(default_factory=list)

    def to_dict(self):
        return {
            "tokens": self.tokens,
            "tokens_text": self.tokens_text,
            "metrics": [asdict(m) for m in self.metrics],
            "sampler_states": [s.name for s in self.sampler_states],
        }
    
def should_stop_branch(token_text, token_context, metrics):
    BRANCH_STOP_TOKENS = {".", "\n", "!", "?", ";", ":", "{", "}", "\n\n", ".\n\n", ":\n\n"}

    if token_text in BRANCH_STOP_TOKENS:
        if token_text == ".":
            # Special handling for ".", check if the previous token is a digit
            if token_context and token_context[-1].isdigit():
                return False  # It's part of a number
        return True
    return False

def feedback(messages: list[Message]):
    MY_API_KEYS = ""
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=MY_API_KEYS)

    completion = client.chat.completions.create(
            # https://openrouter.ai/models
            model="meta-llama/llama-3.1-8b-instruct",
            messages=messages)
    eval = completion.choices[0].message.content
    return eval

def _generate(
    messages: list[Message] | list[dict[str, str]] | str,  # type: ignore -> allow definition to be overriden after type conversion
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    print_stream: bool = False,
    apply_chat_template: bool = True,
    firstgenbranch: bool = True,
    self_feedback: bool = False,
) -> Generator[Tuple[Optional[str], Optional[TokenMetrics], Optional[SamplerState], Optional[GenerationData]], None, None]:
    """
    Core function to generate text using the transformer model and stream the response out.

    Args:
        messages: Input messages or a string prompt
        model: Model to use for generation
        sampler_cfg: Sampler configuration
        max_tokens: Optional, defaults None. Maximum number of tokens to generate, or the max sequence length of the model if None.
        print_stream: Optional, default False. Flag to print the generated tokens to the console
        apply_chat_template: Optional, default True. Flag to apply the chat template to the input messages

    Yields:
        Tuple of (generated token text, token metrics, sampler state, complete Generation object (at the last token only))
    """
    stop_tokens = torch.tensor(model.tokenizer.stop_token_ids, device=device, dtype=torch.int32)
    if max_tokens is None or max_tokens > model.params.max_seq_len: max_tokens = model.params.max_seq_len
    if sampler_cfg is None:
        logging.warning("No sampler config provided, using default config")
        sampler_cfg = SamplerConfig()

    if isinstance(messages, str):
        prompt = messages
        messages = [Message(role="system", content=prompt)]
        logging.warning("entropix.model._generate: prompt passed as a string, cannot save messages to output GenerationData.")
    elif isinstance(messages, list) and isinstance(messages[0], dict):  # convert list[dict] to list[Message] so all messages are validated
        messages = [Message(**m) if not isinstance(m, Message) else m for m in messages]  # type: ignore
    assert isinstance(messages, list) and all(isinstance(m, Message) for m in messages)
    messages: list[Message] = messages  # type: ignore
    if apply_chat_template: prompt = model.tokenizer.apply_chat_template(messages)

    if print_stream:
        print()
        for state, color in STATE_COLOR_MAP.items():
            rprint(f"[{color}]■[/] [dim]{state.value}[/]")
        print()

    with torch.inference_mode():
        tokens = torch.tensor([model.tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')], dtype=torch.long).to(device)
        bs, seqlen = tokens.shape
        cur_pos = 0
        freqs_end = seqlen

        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(model.params.head_dim, model.params.max_seq_len, model.params.rope_theta, model.params.use_scaled_rope)
        kvcache = KVCache.new(model.params.n_layers, bs, model.params.max_seq_len, model.params.n_local_kv_heads, model.params.head_dim).to(device)

        next_token = tokens
        gen_tokens = torch.zeros(1, 1, dtype=torch.int32, device=device)
        response = ""
        gen_tokens_text = []
        gen_logits = []
        gen_metrics = []
        gen_branches = []
        sampler_states = []

        while cur_pos < max_tokens:
            attn = attn_mask if cur_pos < seqlen else None
            logits, kvcache, scores, attn_stats = xfmr(model.weights, model.params, next_token, cur_pos, freqs_cis[cur_pos:freqs_end], kvcache, attn_mask=attn)
            metrics = calculate_metrics(logits, scores)
            next_token, sampler_state = sample(
                logits,
                scores,
                metrics,
                sampler_cfg,
                can_branch=cur_pos >= seqlen,  # NOTE: disallow branching on the first token
                firstgenbranch=firstgenbranch,
            )

            if sampler_state != SamplerState.BRANCHING:
                gen_logits.append(logits)
                gen_metrics.append(metrics)
                sampler_states.append(sampler_state)
                cur_pos = seqlen if cur_pos < seqlen else cur_pos + 1
                freqs_end = cur_pos + 1
                gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                token_text = model.tokenizer.decode([next_token.item()])  # type: ignore (torch.int32 not recognized as int)
                gen_tokens_text.append(token_text)
                if torch.isin(next_token, stop_tokens).any(): break
                response += token_text
                if print_stream: rprint(f"[{STATE_COLOR_MAP[sampler_state]}]{token_text}[/]", end='')
                yield token_text, metrics, sampler_state, None
            else:
                branches = []
                for i, branch_token in enumerate(next_token[0]):
                    branch_token = branch_token.unsqueeze(0)
                    token_text = model.tokenizer.decode([branch_token.item()])  # type: ignore (torch.int32 not recognized as int)
                    prefix = "├─" # if i < len(next_token[0]) - 1 else "└─"
                    if print_stream: rprint(f"\n[{STATE_COLOR_MAP[sampler_state]}]{prefix} {token_text.replace("\n", "\\n")}[/]", end='')
                    branch_pos = cur_pos + 1
                    kvcache = kvcache.cpu()
                    branch_kvcache = copy.deepcopy(kvcache).to(device)
                    branch_gen_logits = [logits]
                    branch_gen_metrics = [metrics]
                    branch_gen_tokens = [branch_token]
                    branch_gen_tokens_text = [token_text]
                    branch_sampler_states = [sampler_state]
                    if not torch.isin(branch_token, stop_tokens).any():
                        while branch_pos < max_tokens:
                            branch_logits, branch_kvcache, branch_scores, _ = xfmr(
                                model.weights, model.params, branch_token, branch_pos, freqs_cis[branch_pos:branch_pos + 1], branch_kvcache, attn_mask=None
                            )
                            branch_gen_logits.append(branch_logits)
                            branch_metrics = calculate_metrics(branch_logits, branch_scores)
                            branch_gen_metrics.append(branch_metrics)
                            branch_token, branch_sampler_state = sample(branch_logits, branch_scores, branch_metrics, sampler_cfg, can_branch=False)
                            branch_gen_tokens.append(branch_token)
                            branch_token_text = model.tokenizer.decode([branch_token.item()])  # type: ignore (torch.int32 not recognized as int)
                            branch_gen_tokens_text.append(branch_token_text)
                            branch_sampler_states.append(branch_sampler_state)
                            branch_pos += 1
                            if print_stream: rprint(f"[{STATE_COLOR_MAP[branch_sampler_state]}]{branch_token_text.replace("\n", "\\n")}[/]", end='')
                            if torch.isin(branch_token, stop_tokens).any() or branch_pos >= max_tokens: break

                            token_context = branch_gen_tokens_text[:-1]
                            stop = should_stop_branch(branch_token_text, token_context, branch_metrics)
                            if stop:
                                break
                            if branch_pos >= max_tokens:
                                break
                    branches.append(
                        Branch(
                            tokens=branch_gen_tokens,
                            kvcache=branch_kvcache.cpu(),
                            cur_pos=branch_pos,
                            tokens_text=branch_gen_tokens_text,
                            metrics=branch_gen_metrics,
                            sampler_states=branch_sampler_states,
                        )
                    )

                gen_branches.append([branch.to_dict() for branch in branches])

                # # TODO: other branch evaluation metrics
                # def score_branch(branch: Branch):
                #     # score = negative average entropy of the branch tokens
                #     entropies = [metrics.logit_entropy for metrics in branch.metrics]
                #     avg_entropy = sum(entropies) / len(entropies)
                #     return -avg_entropy

                # branch_scores = [score_branch(branch) for branch in branches]
                # best_branch_idx = branch_scores.index(max(branch_scores))
                # best_branch = branches[best_branch_idx]

                # Instead of scoring branches by entropy, ask the model itself:
                # Prepare a prompt that lists all candidate completions
                # Before building `analysis_prompt`
                previous_context = "".join(gen_tokens_text)  # or use `response` if you want the full assistant response so far

                analysis_prompt_sys = (
                    "You are an expert evaluator assessing reasoning chains. "
                    "Here're several generated candidate completions below. "
                    "Please choose the most correct and relevant one:\n\n"
                )
                analysis_prompt = ""
                # Optionally, also include original messages for even more context
                for m in messages:
                    analysis_prompt += f"{m.role.upper()}: {m.content}\n"
                analysis_prompt += "\n"

                # Add previously generated tokens for context
                analysis_prompt += "Previously generated tokens:\n" + previous_context + "\n\n"

                # Now list out the candidate branches
                for i, b in enumerate(branches):
                    completion_text = "".join(b.tokens_text)
                    analysis_prompt += f"Option {i}:\n{completion_text}\n\n"

                analysis_prompt += "Which Option number is the best one? Please think step by step shortly then put your final answer in {}. Like {Option X}"

                # Generate the model's response to the analysis prompt
                analysis_messages = [Message(role="system", content=analysis_prompt_sys),
                                    Message(role="user", content=analysis_prompt)]
                if self_feedback:
                    decision_gen = list(_generate(
                        messages=analysis_messages,
                        model=model,
                        sampler_cfg=sampler_cfg,
                        max_tokens=500,
                        print_stream=True,  # Probably don't need to print the decision
                        apply_chat_template=True,
                        firstgenbranch=False,  # Don't allow branching on the decision
                    ))

                    # Parse the model’s final response to find the chosen candidate index
                    # Let's assume the model just returns a number like "0", "1", etc.
                    # decision_gen[-1] should contain the final GenerationData in the 4th element of the tuple
                    # The output of _generate is tuples: (token_text, metrics, sampler_state, GenerationData or None)
                    # The final yield with GenerationData is the last element
                    feedbacks = decision_gen[-1][3]
                    decision_response = feedbacks.response.strip()
                else:
                    feedbacks = feedback(analysis_messages)
                    print(feedbacks)
                    decision_response = feedbacks.strip()

                # Extract the content inside the {}
                match = re.search(r'\{(.*?)\}', decision_response)  # Look for text inside {}
                if match:
                    answer_content = match.group(1).strip()  # Get the text inside {}
                    # Try to extract a number from the content
                    number_match = re.search(r'\b(\d+)\b', answer_content)
                    if number_match:
                        chosen_index = int(number_match.group(1))  # Extract the number
                    else:
                        print("Failed to find a number inside the {}. Defaulting to candidate 0.")
                        chosen_index = 0
                else:
                    print("Failed to find {} in the response. Defaulting to candidate 0.")
                    chosen_index = 0

                best_branch = branches[chosen_index]

                for branch in branches:
                    if branch != best_branch:
                        del branch

                next_token = best_branch.tokens[-1]
                kvcache = best_branch.kvcache.to(device)
                cur_pos = best_branch.cur_pos
                freqs_end = cur_pos + 1
                gen_tokens = torch.cat([gen_tokens, torch.tensor(best_branch.tokens, device=device).unsqueeze(0)], dim=1)
                gen_tokens_text.extend(best_branch.tokens_text)
                gen_metrics.extend(best_branch.metrics)
                sampler_states.extend(best_branch.sampler_states)
                branch_response = "".join(best_branch.tokens_text)
                response += branch_response
                if print_stream:
                    rprint(f"\n[{STATE_COLOR_MAP[SamplerState.BRANCHING]}]╘═══[/]", end='')
                    for state, text in zip(best_branch.sampler_states, best_branch.tokens_text):
                        rprint(f"[{STATE_COLOR_MAP[state]}]{text.replace('\n', '\\n')}[/]", end='')
                if torch.isin(next_token, stop_tokens).any(): break

        if print_stream: print()
        messages.append(Message(role="assistant", content=response))
        gen = GenerationData(
            prompt=prompt,
            response=response,
            tokens=gen_tokens_text,
            messages=messages,
            metrics=gen_metrics,
            sampler_cfg=sampler_cfg,
            sampler_states=sampler_states,
        )
        yield "", metrics, sampler_state, gen

def stream(
    messages: list[Message] | list[dict[str, str]] | str,
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    print_stream: bool = False,
    apply_chat_template: bool = True,
):
    for token_text, metrics, sampler_state, gen in _generate(
        messages=messages,
        model=model,
        sampler_cfg=sampler_cfg,
        max_tokens=max_tokens,
        print_stream=print_stream,
        apply_chat_template=apply_chat_template,
    ):
        yield token_text, metrics, sampler_state, gen

def generate(
    messages: list[Message] | list[dict[str, str]] | str,
    model: Model,
    sampler_cfg: SamplerConfig | None = None,
    max_tokens: int | None = None,
    print_stream: bool = False,
    apply_chat_template: bool = True,
):
    for token_text, metrics, sampler_state, gen in _generate(
        messages=messages,
        model=model,
        sampler_cfg=sampler_cfg,
        max_tokens=max_tokens,
        print_stream=print_stream,
        apply_chat_template=apply_chat_template,
    ):
        if gen is not None:
            return gen
    raise RuntimeError("Generation failed to complete")
