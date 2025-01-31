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
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaModel
import numpy as np
from transformers import PreTrainedModel
import itertools
from tqdm import trange
import emoji
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)


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
        zeroembed_adapter = None,
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
        
        
        if zeroembed_adapter is not None:
            if not hasattr(zeroembed_adapter,"inputs_embeds"):
                inputs_embeds.retain_grad()
                zeroembed_adapter.inputs_embeds = inputs_embeds

            inputs_embeds = zeroembed_adapter(input_ids, inputs_embeds)
            

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

    def _forward(self, attn_weights: torch.Tensor):
        self.attn_weights = attn_weights
        self.attn_weights.retain_grad()
        return self.attn_weights

    @property
    def grad(self):
        return self.attn_weights.grad

    @property
    def weight(self):
        return self.attn_weights
    
    @property
    def saliency(self):
        return self.attn_weights * self.attn_weights.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.attn_weights.grad is not None:
            if set_to_none:
                self.attn_weights.grad = None
            else:
                self.attn_weights.grad.zero_()

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



class ZeroEmbedAdapter(nn.Module):
    def __init__(self, nonzero_poss, factor):
        super().__init__()
        self.nonzero_poss = sorted(set(nonzero_poss))
        # print("poss:",self.nonzero_poss)
        self.factor = factor
    

    def forward(self, input_ids, input_embeds):
        # print("input_length:",input_ids.shape, input_embeds.shape)
        # poss = list(range(input_ids.size(-1)))
        if self.inputs_embeds.grad is None:

            print("Grad None!")
            return input_embeds
        print("Grad Not None!", self.factor)
        # fac_poss = torch.zeros_like(input_embeds) + self.factor * self.inputs_embeds.grad
        
        fac_poss = torch.zeros_like(input_embeds) 
        # print("poss:",self.nonzero_poss)
        print(fac_poss.shape)
        for span_idx in self.nonzero_poss:
            fac_poss[:, span_idx[0]:span_idx[1]] = \
                self.factor * self.inputs_embeds.grad[:, span_idx[0]:span_idx[1]]

        # print("into zero:",(torch.abs(fac_poss-1.)<1e-5).sum(), sum((s[1]-s[0] for s in self.nonzero_poss), 0))
        # print(fac_poss.flatten().tolist().count(1.0))
        self.zero_grad()

        self.grad_input_embeds = input_embeds - fac_poss
        return self.grad_input_embeds

    def zero_grad(self):
        self.inputs_embeds.grad.zero_()


class ZeroEmbedManager:
    def __init__(self, model: PreTrainedModel, model_name: str,
                 nonzero_poss,
                 factor 
                 ):
        self.model = model
        self.model_name = model_name
        # self.attention_adapters = self.register_attentioner_to_model()
        self.zeroembed_adapter = ZeroEmbedAdapter(nonzero_poss, factor)

        self.model_origin_forward = self.model.model.forward

        self.model.model.forward = partial(hack_forward_llama, 
                                           self.model.model, 
                                           zeroembed_adapter = self.zeroembed_adapter)
    
    def zero_grad(self):
        self.zeroembed_adapter.zero_grad()
    
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
        topk_values ,topk_indices = np_topk(saliency[target_pos, :], 20)
        topk_values = torch.tensor(topk_values).flatten()
        topk_indices = torch.tensor(topk_indices).flatten()
        values[topk_indices] += topk_values
    
    return np_topk(values.numpy(), k)


def cal_temp(saliency, target_poss, span_ids):
    temp = 0
    length = 0
    for span_idx in span_ids:
        for target_pos in range(*target_poss):
            temp += saliency[target_pos, np.array(range(span_idx[0], span_idx[1]))].sum()

        length += (span_idx[1] - span_idx[0])
    return temp/(target_poss[1] - target_poss[0]), length


    

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


def test_model_with_attention_adapter(model, input, golden, search_pos, attack_pos, emoji_pos, target_poss, is_0k, model_name, tokenizer, take_last_loss = True, with_adapter=False, start_layer = 0, factor = 0.1):
    """
    zecheng_note: 这里计算的是language modeling loss    
    """
    zeroembed_manager = ZeroEmbedManager(model, model_name,
                     nonzero_poss=search_pos + attack_pos,
                     factor = factor)

    output = model(input)
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

    output = model(input)

    return model, zeroembed_manager.zeroembed_adapter.grad_input_embeds, zeroembed_manager.model_origin_forward
    


def random_combine(ref:list, att:list, return_snd_pos = False, seed = None):
    if seed is not None:
        random.seed(seed)
    att_list =[[] for _ in range(len(ref) + 1)]
    for p_att in att[:-1]:
        att_list[random.randint(0,len(ref)-1)].append(p_att)
    att_list[-1].append(att[-1])

    results = [k for k in att_list[0]]

    if return_snd_pos:
        insert_pos = list(range(len(results)))
    for r, patt in zip(ref,att_list[1:]):
        results.append(r)
        if return_snd_pos:
            insert_pos.extend(list(range(len(results), len(results) + len(patt))))

        results.extend(patt)
            
    if return_snd_pos:
        assert len(att) == len(insert_pos)
        return results, insert_pos

    return results


def get_random_emoji(tokenizer, num=50, return_idx=True, seed=None):
    all_emojis = list(emoji.EMOJI_DATA.keys())  # get all emojis
    if seed is not None:
        random.seed(seed)
    random_emojis = random.sample(all_emojis, num)
    print(f"your chose emoji: {random_emojis}")
    if return_idx:
        index_emojis = []
        for e in random_emojis:
            index_emojis.append(tokenizer(e, add_special_tokens=False).input_ids)
        return index_emojis
    return random_emojis

def begin_test(args, question, answer, selected_idx, model, tokenizer, depth_percent, background_text, disturb_pos,disturb_tok_needles, evidence, evidence_list, save_file_name, model_name, is_0k, use_emoji, with_adapter=False, start_layer = 0, factor = 0.01):
    if background_text is not None:

        if use_emoji:
            emojis10 = get_random_emoji(tokenizer, 10, return_idx = True, seed = 42)
            background_text, emoji_pos = random_combine(background_text, emojis10, 
                                                         return_snd_pos = True, seed = 42)
            emoji_pos = set(emoji_pos)
            cumsum_num = 0
            emoji_spans = []

        depth_percent = [i / 10 for i in depth_percent]
        updated_sample = [[] for _ in range(len(background_text) + 1)]
        real_pos = [int(len(background_text) * i) for i in depth_percent]
        for fact, pos in zip(evidence, real_pos):  # insert real needle
            updated_sample[pos].append(fact)
        for fact, pos in zip(disturb_tok_needles, disturb_pos):  # insert disturb needle
            updated_sample[pos].append(fact)


    
        for i, s in enumerate(background_text):  # insert irrevelent needle
            if use_emoji and (i in emoji_pos):

                cur_pos = sum((len(l) for l in updated_sample[i]), 0)
                emoji_spans +=[(cumsum_num + cur_pos, cumsum_num + cur_pos + len(s))]

            updated_sample[i].append(s)

            if use_emoji:
                cumsum_num += sum((len(l) for l in updated_sample[i]), 0)
    else:
        updated_sample = random_combine(evidence[:-1], disturb_tok_needles+[evidence[-1]], seed = 42)
        updated_sample = [[k] for k in updated_sample]
        # print("updated_sample:", updated_sample)
    
    if not use_emoji or is_0k:
        emoji_spans = []
        

    flat = [i for s in updated_sample for i in s]
    tokens = [i for s in flat for i in s]

    new_context = tokenizer.decode(tokens)
    input_context = new_context + f"\n{question}\nAnswer:"
    if tokenizer.chat_template is not None:
        shift = 30
        inp = tokenizer.apply_chat_template([{ "role": "user", "content": input_context}], tokenize=True, add_generation_prompt=True, return_tensors='pt')
    else:
        shift = 0
        inp = tokenizer(input_context, return_tensors='pt').input_ids
    emoji_spans = [(k[0] + shift, k[1] + shift) for k in emoji_spans]

    

    search_pos = find_multi_needle_idx(inp[0], tokenizer, evidence_list[selected_idx])
    attack_pos = find_multi_needle_idx(inp[0], tokenizer, disturb_tok_needles)
    inp = inp.to(model.device)
    
    
    print("ref:",search_pos)
    print("att:",attack_pos)
    print("emj:",emoji_spans)
    # return

    with torch.no_grad():
        pred_res = tokenizer.decode(model.generate(inp, max_new_tokens=32, do_sample=False)[0, inp.size(-1):])
        logger.info(pred_res)

    logger.info(inp.shape)

    if tokenizer.chat_template is not None:
        inp = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_context}, {"role": "assistant", "content": answer}], 
            tokenize=True, add_generation_prompt=False, return_tensors='pt'
        ).to(model.device)
    else:
        inp = tokenizer(input_context + "\n" + answer, return_tensors='pt').input_ids.to(model.device)
    

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

        model, input_embeds, origin_forward = test_model_with_attention_adapter(model, inp, label, search_pos, attack_pos, 
                                                     emoji_spans,
                                                     (target_pos[0] - 1,
                                                      target_pos[1] - 1), 
                                                      is_0k,
                                                      model_name, tokenizer, with_adapter=with_adapter,
                                                      start_layer = start_layer,
                                                      factor = factor)
    
    print("NEW@!")
    # input_embeds = input_embeds.cpu()
    # del model
    # torch.cuda.empty_cache()
    with torch.no_grad():
        # model = AutoModelForCausalLM.from_pretrained(args.model_path, 
        #                                                                 attn_implementation = "flash_attention_2"
        #                                                                 ).half().eval()
        model.model.forward = origin_forward
        print("valid:", inp.shape, input_embeds.shape)
        output = model.eval().generate(inputs_embeds = input_embeds.to(model.device),
                                                    #  max_length=32, 
                                                     max_new_tokens=32, 
                                                    do_sample=False,
                                                    pad_token_id = tokenizer.eos_token_id,
                                                     )
        # print("output:",output)
        pred_res_z = tokenizer.decode(output[0, :])
        # pred_res_z = "none"
        logger.info("pred:")
        logger.info(pred_res_z)
    flow_res = {}
    

    flow_res["pred_res_origin"] = pred_res.lower()
    flow_res["pred_res"] = pred_res_z.lower()
    flow_res["answer"] = answer.lower() 
    flow_res["score"] = 100 if answer.lower() in pred_res_z.lower() else 0

    logger.info(flow_res)
    auto_save_data(flow_res, f"{args.save_dir}/{save_file_name}.pkl")
