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
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
import numpy as np
from transformers import PreTrainedModel
import itertools
from tqdm import trange
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
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




def hack_forward_llama(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        self.recorded_current_input_ids = input_ids
        self.recorded_current_embeds = inputs_embeds
        self.recorded_current_embeds.retain_grad()
        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class EmbeddingManager:
    def __init__(self, model: PreTrainedModel, model_name: str):
        self.model = model
        self.model_name = model_name
        
        self.model.model.forward = partial(hack_forward_llama, self.model.model)
    
    @property
    def current_input_ids(self):
        return self.model.model.recorded_current_input_ids

    @property
    def current_embeds(self):
        return self.model.model.recorded_current_embeds


    def zero_grad(self,set_to_none=True):
        
        if set_to_none:
            self.current_embeds.grad = None
        else:
            self.current_embeds.grad.zero_()

    def grad_process(self, grad, use_abs = True):
        assert len(grad.shape) == 3
        if use_abs:
            grad = abs(grad)

        grad = grad.sum(-1)
        return grad

    def grad(self,*args,**kwargs):
        grad = self.current_embeds * self.current_embeds.grad
        return self.grad_process(grad, *args, **kwargs)
        



# class AttentionerManager(AttentionerManagerBase):
#     def __init__(self, model: PreTrainedModel, attention_adapters: List[AttentionAdapterBase]):
#         super().__init__(model, attention_adapters)

#     def register_attentioner_to_model(self):
#         for i, layer in enumerate(self.model.model.layers):
#             # layer.config = self.model.config
#             layer.self_attn.forward = partial(hack_attn, layer.self_attn, attention_adapter=self.attention_adapters[i])

    
    
def np_topk(arr, k):
    # 获取排序后的索引（从小到大）
    sorted_indices = np.argsort(arr)
    # 获取前k个最大元素的索引
    topk_indices = sorted_indices[-k:]
    # 获取前k个最大元素的值
    topk_values = arr[topk_indices]
    return topk_values, topk_indices


def multi_torch_topk(saliency, target_poss, k):
    values = torch.full((saliency.shape[-1],), 0.)
    for target_pos in range(*target_poss):
        topk_values ,topk_indices = np_topk(saliency[target_pos, :],20)
        topk_values = torch.tensor(topk_values).flatten()
        topk_indices = torch.tensor(topk_indices).flatten()
        values[topk_indices] += topk_values
    
    return np_topk(values.numpy(), k)


def calculate_portions(saliency, evi_poss: List[Tuple[int, int]], attack_pos: List[Tuple[int, int]], target_poss: Tuple[int, int]):
    """
    saliency: [batch_size, seq_len, seq_len] 倒数第二个位置对应prediction token

    target_poss: [l, r)
    """
    saliency = saliency.float().detach().clone().cpu()
    assert len(saliency.shape) == 2 or (len(saliency.shape) == 3 and saliency.shape[0] == 1)
    if len(saliency.shape) == 3:
        saliency = saliency.squeeze(0)
        
    saliency = saliency.numpy() #(seq_len, seq_len)
    np.fill_diagonal(saliency, 0)
    total_context_length = saliency.shape[1]

    # zecheng_note: 查询信息流里面的Peak Tokens, 关键词Flow
    # _, topk_indices = np_topk(saliency[target_poss, :], 20)
    
    _, topk_indices = multi_torch_topk(saliency, target_poss, 20)

    # add: proportion-n: each evidence -> target token
    evidence_proportions = []

    
    # proportion1: evidence -> target token (zecheng_note: 需要被查询的位置放前面)
    proportion1 = 0
    evidence_length = 0
    for span_idx in evi_poss:
        temp_proportion1 = 0
        for target_pos in range(*target_poss):
            temp_proportion1 += saliency[target_pos, np.array(range(span_idx[0], span_idx[1]))].sum()
        proportion1 += temp_proportion1/(target_poss[1] - target_poss[0])
        
        # evidence proportions
        evidence_length += span_idx[1] - span_idx[0]
        
        temp_evidence_length = 0
        for target_pos in range(*target_poss):
            temp_evidence_length += saliency[target_pos, np.array(range(span_idx[0], span_idx[1]))].sum() / (span_idx[1] - span_idx[0])
        
        evidence_proportions.append(temp_evidence_length/(target_poss[1] - target_poss[0]))

    # proportion2: all context -> target token
    temp_proportion2 = 0
    for target_pos in range(*target_poss):
        temp_proportion2 +=saliency[target_pos, :].sum()
    proportion2 = temp_proportion2/(target_poss[1] - target_poss[0])

    # proportion3: irrevelent evidence -> target token
    proportion3 = 0
    irr_evidence_length = 0
    for span_idx in attack_pos:
        temp_proportion3 = 0
        for target_pos in range(*target_poss):
            temp_proportion3 += saliency[target_pos, np.array(range(span_idx[0], span_idx[1]))].sum()

        proportion3 += temp_proportion3/(target_poss[1] - target_poss[0])
        
        irr_evidence_length += span_idx[1] - span_idx[0]

    # proportion4: remain context -> target token
    proportion4 = proportion2 - proportion1 - proportion3

    proportion1 = proportion1 / evidence_length if evidence_length != 0 else 0  # zecheng note: evidence 长度可能会为0
    proportion2 = proportion2 / total_context_length
    proportion3 = proportion3 / irr_evidence_length if irr_evidence_length != 0 else 0 # zecheng note: irr_evidence_length 长度可能会为0
    proportion4 = proportion4 / (total_context_length - evidence_length - irr_evidence_length)

    return proportion1, proportion2, proportion3, proportion4, evidence_proportions, topk_indices

    # proportion1: evidence -> target token (zecheng_note: 需要被查询的位置放前面)
    proportion1 = 0
    evidence_length = 0
    for span_idx in evi_poss:
        proportion1 += saliency[target_poss, np.array(range(span_idx[0], span_idx[1]))].sum()
        evidence_length += span_idx[1] - span_idx[0]
        # evidence proportions
        evidence_proportions.append(saliency[target_poss, np.array(range(span_idx[0], span_idx[1]))].sum() / (span_idx[1] - span_idx[0]))

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
    

def calculate_embedding_portions(saliency, 
                                 evi_poss: List[Tuple[int, int]], 
                                 attack_pos: List[Tuple[int, int]], 
                                 target_poss: Tuple[int, int]):
    """
    saliency: [batch_size, seq_len] 

    target_poss: [l, r)
    """
    saliency = saliency.float().detach().clone().cpu()
    assert len(saliency.shape) == 1 or (len(saliency.shape) == 2 and saliency.shape[0] == 1)
    if len(saliency.shape) == 2:
        saliency = saliency.squeeze(0)
        
    saliency = saliency.numpy() #(seq_len, )

    total_context_length = saliency.shape[0]

    
    _, topk_indices = np_topk(saliency, 20)# NOTE:  embedding 分数最高的20个词

    # add: proportion-n: each evidence -> target token
    evidence_proportions = []
    
    # proportion1: evidence -> target token (zecheng_note: 需要被查询的位置放前面)
    proportion1 = 0 # NOTE: proportion1: evidence token
    evidence_length = 0
    for span_idx in evi_poss:
        temp = saliency[np.array(range(*span_idx))].sum()
        proportion1 += temp
        # evidence proportions
        evidence_length += (span_idx[1] - span_idx[0])

        evidence_proportions += [temp/(span_idx[1] - span_idx[0])]

    # proportion2: all context -> target token
    proportion2 = saliency.sum()

    # proportion3: irrevelent evidence -> target token
    proportion3 = 0
    irr_evidence_length = 0
    for span_idx in attack_pos:
        proportion3 += saliency[np.array(range(*span_idx))].sum()
        irr_evidence_length += (span_idx[1] - span_idx[0])

    # proportion4: remain context -> target token
    proportion4 = proportion2 - proportion1 - proportion3

    proportion1 = proportion1 / evidence_length if evidence_length != 0 else 0  # zecheng note: evidence 长度可能会为0
    proportion2 = proportion2 / total_context_length
    proportion3 = proportion3 / irr_evidence_length if irr_evidence_length != 0 else 0 # zecheng note: irr_evidence_length 长度可能会为0
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


def test_model_embedding(model, input, golden, search_pos, attack_pos, target_poss, model_name, tokenizer, 
                         take_last_loss = True):
    """
    zecheng_note: 这里计算的是language modeling loss    
    """
    embeddingmanger = EmbeddingManager(model, model_name)
    output = model(input)
    embeddingmanger.zero_grad()
    # print("input_golden_size:", input.size(),golden.size())
    if input.size(-1) == golden.size(-1):
        logits = output['logits'][:, :-1, :]
        labels = golden[:, 1:]
    else:
        logits = output['logits'][:, -1, :]
        labels = golden[:, -1]
    # print("logits_shape:",logits.shape, labels)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()

    pros_dict = dict()
    saliency = embeddingmanger.grad(use_abs=True)

    proportion1, proportion2, proportion3, proportion4, evidence_proportions, topk_indices =\
          calculate_embedding_portions(saliency, search_pos, attack_pos, target_poss)
    top_tokens = []
    for idx in topk_indices:
        top_tokens.append(tokenizer.decode(input[0][idx].item()))

    pros_dict['embedding'] = {'score': [proportion1, proportion2, proportion3, proportion4], 
                              "topk_indices": topk_indices,
                              'topk_tokens': top_tokens,'evidence_proportions': evidence_proportions}
    
    return pros_dict

    for i in trange(attentionermanger.start_layer, len(attentionermanger.attention_adapters)):
        saliency = saliencies[i-attentionermanger.start_layer]        
        proportion1, proportion2, proportion3, proportion4, evidence_proportions, topk_indices = calculate_portions(saliency, search_pos, attack_pos, target_poss)
        top_tokens = []
        for idx in topk_indices:
            top_tokens.append(tokenizer.decode(input[0][idx].item()))
        pros_dict[i] = {'score': [proportion1, proportion2, proportion3, proportion4], 'topk_tokens': top_tokens, 'evidence_proportions': evidence_proportions}

    return pros_dict

def random_combine(ref:list, att:list):

    att_list =[[] for _ in range(len(ref) + 1)]
    for p_att in att[:-1]:
        att_list[random.randint(0,len(ref)-1)].append(p_att)
    att_list[-1].append(att[-1])

    results = [k for k in att_list[0]]

    for r, patt in zip(ref,att_list[1:]):
        results.append(r)
        results.extend(patt)

    return results


def begin_test(args, question, answer, selected_idx, model, tokenizer, depth_percent, background_text, disturb_pos,disturb_tok_needles, evidence, evidence_list, save_file_name, model_name, with_adapter=False, start_layer = 0):
    if background_text is not None:
        depth_percent = [i / 10 for i in depth_percent]
        updated_sample = [[] for _ in range(len(background_text) + 1)]
        real_pos = [int(len(background_text) * i) for i in depth_percent]
        for fact, pos in zip(evidence, real_pos):  # insert real needle
            updated_sample[pos].append(fact)
        for fact, pos in zip(disturb_tok_needles, disturb_pos):  # insert disturb needle
            updated_sample[pos].append(fact)
        for i, s in enumerate(background_text):  # insert irrevelent needle
            updated_sample[i].append(s)
    else:
        updated_sample = random_combine(evidence[:-1], disturb_tok_needles+[evidence[-1]])
        updated_sample = [[k] for k in updated_sample]
        # print("updated_sample:", updated_sample)

    flat = [i for s in updated_sample for i in s]
    tokens = [i for s in flat for i in s]

    new_context = tokenizer.decode(tokens)
    input_context = new_context + f"\n{question}\nAnswer:"
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
            {"role": "assistant", "content": answer}
        ], 
        tokenize=True, add_generation_prompt=False, return_tensors='pt'
    ).to(model.device)
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors='pt')["input_ids"].to(model.device)
    toks_length = answer_ids.size(-1)
    for j in range(inp.size(-1), toks_length, -1):
        if (inp[0, j-toks_length : j] == answer_ids).sum().item() == toks_length:
            target_pos = (j-toks_length, j) 
            break
    else:
        raise ValueError("Not find target in input tokens!")
    # print("ANDADSDSAD:", inp.shape, target_pos, answer_ids, toks_length,answer)
    
    if args.loss_type == "label":
        label = torch.full(inp.shape, -100).to(model.device)
        for sub_pos in range(*target_pos):
            label[0, sub_pos] = inp[0, sub_pos]

        flow_res = test_model_embedding(model, inp, label, search_pos, attack_pos, 
                                                     (target_pos[0] - 1,
                                                      target_pos[1] - 1), 
                                                      model_name, tokenizer)
    
    elif args.loss_type == "ce":
         # shift to left before the label token
        flow_res = test_model_embedding(model, inp, inp, search_pos, attack_pos, 
                                                     (target_pos[0] - 1,
                                                      target_pos[1] - 1), 
                                                      model_name, tokenizer)

    flow_res["pred_res"] = pred_res
    flow_res["score"] = 100 if answer.lower() in pred_res.lower() else 0

    logger.info(flow_res)
    auto_save_data(flow_res, f"{args.save_dir}/{save_file_name}.pkl")