from modelzipper.tutils import *
import sys
import os
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.info(sys.path)
from retrieval_head_detection import SentenceSampler
import torch.nn.functional as F
import datasets
from functools import wraps, partial
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from baukit import Trace
import numpy as np
from transformers import PreTrainedModel
import itertools
from tqdm import trange


def hack_attn_phi(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    attention_adapter=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    if self.qk_layernorm:
        query_states = self.q_layernorm(query_states)
        key_states = self.k_layernorm(key_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings

    # Partial rotary embedding
    query_rot, query_pass = (
        query_states[..., : self.rotary_ndims],
        query_states[..., self.rotary_ndims :],
    )
    key_rot, key_pass = (
        key_states[..., : self.rotary_ndims],
        key_states[..., self.rotary_ndims :],
    )
    # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
    query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

    # [batch_size, seq_length, num_heads, head_dim]
    query_states = torch.cat((query_rot, query_pass), dim=-1)
    key_states = torch.cat((key_rot, key_pass), dim=-1)

    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "partial_rotation_size": self.rotary_ndims,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
    attn_weights = torch.matmul(
        query_states.to(torch.float32), key_states.to(torch.float32).transpose(2, 3)
    ) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights += causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)

    if attention_adapter is not None:  # pass attention weights to adapter
        attn_weights = attention_adapter(attn_weights)

    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.dense(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def hack_attn_qwen(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_adapter=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    if attention_adapter is not None:  # pass attention weights to adapter
        attn_weights = attention_adapter(attn_weights)

    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def hack_attn_llama(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_adapter=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype) # FIXME: use bf16 to save memory

    if attention_adapter is not None:  # pass attention weights to adapter
        attn_weights = attention_adapter(attn_weights)
    
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



class AttentionAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def forward(self, attn_weights):
        if self.use_flag:
            return self._forward(attn_weights)
        else:
            return attn_weights

    def _forward(self, attn_weights):
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class AttentionAdapter(AttentionAdapterBase):
    def __init__(self) -> None:
        super().__init__()
        self.params = None

    def _forward(self, attn_weights):
        if self.params is None:
            self.params = torch.ones_like(attn_weights, requires_grad=True)
        else:
            self.params.data = torch.ones_like(attn_weights)
        return attn_weights * self.params

    @property
    def grad(self):
        return self.params.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.params.grad is not None:
            if set_to_none:
                self.params.grad = None
            else:
                self.params.grad = torch.zeros_like(self.params.grad)

def manager_decoractor(manager):
    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            return fn(*args, **kwargs)

        return wrapper

    return model_forward_decorator



class AttentionerManagerBase:
    def __init__(self, model: PreTrainedModel, model_name: str):
        self.model = model
        self.model_name = model_name
        self.attention_adapters = self.register_attentioner_to_model()
        self.model.forward = manager_decoractor(self)(self.model.forward)

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        for attention_adapter in self.attention_adapters:
            attention_adapter.register_input_ids(input_ids)

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_attentioner_to_model(self):
        raise NotImplementedError

    def zero_grad(self,set_to_none=True):
        if set_to_none:
            for attention_adapter in self.attention_adapters:
                attention_adapter.params = None
        else:
            for attention_adapter in self.attention_adapters:
                attention_adapter.zero_grad(set_to_none=True)

    def grad_process(self, grad,use_abs = True):
        assert len(grad.shape) == 4
        grad = grad.sum(1)
        if use_abs:
            grad = abs(grad)
        return grad

    def grad(self,*args,**kwargs):
        grads = []
        for attention_adapter in self.attention_adapters:
            grads.append(self.grad_process(attention_adapter.params.grad,*args,**kwargs))
        return grads


# class AttentionerManager(AttentionerManagerBase):
#     def __init__(self, model: PreTrainedModel, attention_adapters: List[AttentionAdapterBase]):
#         super().__init__(model, attention_adapters)

#     def register_attentioner_to_model(self):
#         for i, layer in enumerate(self.model.model.layers):
#             # layer.config = self.model.config
#             layer.self_attn.forward = partial(hack_attn, layer.self_attn, attention_adapter=self.attention_adapters[i])


class AttentionerManager(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel, model_name: str):
        super().__init__(model, model_name)
        self.model_name = model_name
    def register_attentioner_to_model(self):
        attention_adapters = []
        for i, layer in enumerate(self.model.model.layers):
            attention_adapter = AttentionAdapter()
            if "llama" in self.model_name.lower():
                layer.self_attn.forward = partial(hack_attn_llama, layer.self_attn, attention_adapter=attention_adapter)
            elif "qwen" in self.model_name.lower():
                layer.self_attn.forward = partial(hack_attn_qwen, layer.self_attn, attention_adapter=attention_adapter)
            else:
                raise NotImplementedError(f"{self.model_name} not supported")
            attention_adapters.append(attention_adapter)
        return attention_adapters
    
def np_topk(arr, k):
    # 获取排序后的索引（从小到大）
    sorted_indices = np.argsort(arr)
    # 获取前k个最大元素的索引
    topk_indices = sorted_indices[-k:]
    # 获取前k个最大元素的值
    topk_values = arr[topk_indices]
    return topk_values, topk_indices


def calculate_portions(saliency, evi_poss: List[Tuple[int, int]], attack_pos: List[Tuple[int, int]], target_poss: int):
    """
    saliency: [batch_size, seq_len, seq_len] 倒数第二个位置对应prediction token
    """
    saliency = saliency.float().detach().clone().cpu()
    assert len(saliency.shape) == 2 or (len(saliency.shape) == 3 and saliency.shape[0] == 1)
    if len(saliency.shape) == 3:
        saliency = saliency.squeeze(0)
    saliency = saliency.numpy()
    np.fill_diagonal(saliency, 0)
    total_context_length = saliency.shape[1]

    # zecheng_note: 查询信息流里面的Peak Tokens, 关键词Flow
    _, topk_indices = np_topk(saliency[target_poss, :], 20)

    # add: proportion-n: each evidence -> target token
    evidence_proportions = []

    # proportion1: evidence -> target token (zecheng_note: 需要被查询的位置放前面)
    proportion1 = 0
    evidence_length = 0
    for span_idx in evi_poss:
        proportion1 += saliency[target_poss, np.array(range(span_idx[0], span_idx[1]))].sum()
        evidence_length += span_idx[1] - span_idx[0]
        # evidence proportions
        evidence_proportions.append(saliency[target_poss, np.array(range(span_idx[0], span_idx[1]))].sum() // (span_idx[1] - span_idx[0]))

    # proportion2: all context -> target token
    proportion2 = saliency[target_poss, :].sum()

    # proportion3: irrevelent evidence -> target token
    proportion3 = 0
    irr_evidence_length = 0
    for span_idx in attack_pos:
        proportion3 += saliency[target_poss, np.array(range(span_idx[0], span_idx[1]))].sum()
        irr_evidence_length += span_idx[1] - span_idx[0]

    # proportion4: remain context -> target token
    proportion4 = proportion2 - proportion1 - proportion3

    proportion1 = proportion1 / evidence_length
    proportion2 = proportion2 / total_context_length
    proportion3 = proportion3 / irr_evidence_length
    proportion4 = proportion4 / (total_context_length - evidence_length - irr_evidence_length)

    return proportion1, proportion2, proportion3, proportion4, evidence_proportions, topk_indices
    


def get_proportion_wla(saliency, class_poss, final_poss):
    saliency = saliency.detach().clone().cpu()
    class_poss = torch.hstack(class_poss).detach().clone().cpu()
    final_poss = final_poss.detach().clone().cpu()
    assert len(saliency.shape) == 2 or (len(saliency.shape) == 3 and saliency.shape[0] == 1)
    if len(saliency.shape) == 3:
        saliency = saliency.squeeze(0)
    saliency = saliency.numpy()
    np.fill_diagonal(saliency, 0)
    proportion1 = saliency[class_poss, :].sum()
    proportion2 = saliency[final_poss, class_poss].sum()
    proportion3 = saliency.sum() - proportion1 - proportion2

    N = int(final_poss)
    sum3 = (N + 1) * N / 2 - sum(class_poss) - len(class_poss)
    proportion1 = proportion1 / sum(class_poss)
    proportion2 = proportion2 / len(class_poss)
    proportion3 = proportion3 / sum3
    proportions = np.array([proportion1, proportion2, proportion3])
    return proportions





def find_multi_needle_idx(input_ids, tokenizer, needles):
    all_evi_pos = []
    for i, evi in enumerate(needles):
        if isinstance(evi, str):
            needle_ids = tokenizer(evi, add_special_tokens=False)["input_ids"]
        else:
            needle_ids = evi
        logger.info(f"evidence {i} --> {tokenizer.decode(needle_ids, skip_special_tokens=False)}")
        span_len = len(needle_ids)
        for j in range(len(input_ids)):
            token_span = input_ids[j : j + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if(overlap > 0.8):
                all_evi_pos.append((j + 1, j + span_len))
                logger.info(f"find evidence {i} at --> {(j + 1, j + span_len)} --> {tokenizer.decode(input_ids[j + 1: j + span_len], skip_special_tokens=False)}")
                break
    return all_evi_pos


def test_model_with_attention_adapter(model, input, golden, search_pos, attack_pos, target_poss, model_name, tokenizer, take_last_loss = True):
    """
    zecheng_note: 这里计算的是language modeling loss
    """
    attentionermanger = AttentionerManager(model, model_name)
    attentionermanger.zero_grad()

    output = model(input)
    
    if input.size(-1) == golden.size(-1):
        logits = output['logits'][:, :-1, :]
        labels = golden[:, 1:]
    else:
        logits = output['logits'][:, -1, :]
        labels = golden[:, -1]

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()

    pros_dict = dict()
    for i in trange(len(attentionermanger.attention_adapters)):
        saliency = attentionermanger.grad(use_abs=True)[i]        
        proportion1, proportion2, proportion3, proportion4, evidence_proportions, topk_indices = calculate_portions(saliency, search_pos, attack_pos, target_poss)
        top_tokens = []
        for idx in topk_indices:
            top_tokens.append(tokenizer.decode(input[0][idx].item()))
        pros_dict[i] = {'score': [proportion1, proportion2, proportion3, proportion4], 'topk_tokens': top_tokens, 'evidence_proportions': evidence_proportions}
    return pros_dict


def begin_test(args, question, answer, selected_idx, model, tokenizer, depth_percent, background_text, disturb_pos,disturb_tok_needles, evidence, evidence_list, save_file_name, model_name):

    depth_percent = [i / 10 for i in depth_percent]
    updated_sample = [[] for _ in range(len(background_text) + 1)]
    real_pos = [int(len(background_text) * i) for i in depth_percent]
    for fact, pos in zip(evidence, real_pos):  # insert real needle
        updated_sample[pos].append(fact)
    for fact, pos in zip(disturb_tok_needles, disturb_pos):  # insert disturb needle
        updated_sample[pos].append(fact)
    for i, s in enumerate(background_text):  # insert irrevelent needle
        updated_sample[i].append(s)

    flat = [i for s in updated_sample for i in s]
    tokens = [i for s in flat for i in s]

    new_context = tokenizer.decode(tokens)
    input_context = new_context + f"\nQuestion: {question}\nAnswer:"
    inp = tokenizer.apply_chat_template([{ "role": "user", "content": input_context}], tokenize=True, add_generation_prompt=True, return_tensors='pt')

    search_pos = find_multi_needle_idx(inp[0], tokenizer, evidence_list[selected_idx])
    attack_pos = find_multi_needle_idx(inp[0], tokenizer, disturb_tok_needles)
    inp = inp.to(model.device)
    
    with torch.no_grad():
        pred_res = tokenizer.decode(model.generate(inp, max_new_tokens=32, do_sample=False)[0, inp.size(-1):])
        logger.info(pred_res)

    logger.info(inp.shape)

    inp = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": input_context}, 
            {"role": "assitant", "content": answer}
        ], 
        tokenize=True, add_generation_prompt=False, return_tensors='pt'
    ).to(model.device)
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors='pt')["input_ids"].to(model.device)
    
    for j in range(inp.size(-1), 0, -1):
        if inp[0, j - 1] == answer_ids:
            target_pos = j - 1 
    
    if args.loss_type == "label":
        label = torch.full(inp.shape, -100).to(model.device)
        label[0, target_pos] = inp[0, target_pos]
        flow_res = test_model_with_attention_adapter(model, inp, label, search_pos, attack_pos, target_pos-1, model_name, tokenizer)
    
    elif args.loss_type == "ce":
         # shift to left before the label token
        flow_res = test_model_with_attention_adapter(model, inp, inp, search_pos, attack_pos, target_pos-1, model_name, tokenizer)

    flow_res["pred_res"] = pred_res
    flow_res["score"] = 100 if answer.lower() in pred_res.lower() else 0

    logger.info(flow_res)
    auto_save_data(flow_res, f"{args.save_dir}/{save_file_name}.pkl")



