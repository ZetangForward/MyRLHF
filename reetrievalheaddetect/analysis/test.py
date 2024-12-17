from modelzipper.tutils import *
import sys
import os
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
from retrieval_head_detection import SentenceSampler
import torch.nn.functional as F
import datasets
from functools import wraps, partial
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from baukit import Trace
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


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
    def __init__(self, model):
        self.model = model
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


def hack_attn(
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



class AttentionerManager(AttentionerManagerBase):
    def __init__(self, model):
        super().__init__(model)

    def register_attentioner_to_model(self):
        attention_adapters = []
        for i, layer in enumerate(self.model.transformer.h):
            attention_adapter = AttentionAdapter()
            layer.attn._attn = partial(hack_attn, layer.self_attn,
                                       attention_adapter=attention_adapter)
            attention_adapters.append(attention_adapter)
        return attention_adapters


def get_proportion(saliency, class_poss, final_poss):
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


def test_model_with_attention_adapter(model, input, golden):
    attentionermanger = AttentionerManager(model)
    attentionermanger.zero_grad()
    import ipdb; ipdb.set_trace()
    output = attentionermanger.forward(input)
    loss = F.cross_entropy(output['logits'], golden)
    loss.backward()

    pros = []
    for i in range(len(attentionermanger.attention_adapters)):
        saliency = attentionermanger.grad(use_abs=True)[i]
        import ipdb; ipdb.set_trace()
    #     pro = get_proportion(saliency, class_poss, final_poss)
    #     pros.append(pro)
    # pros = np.array(pros)
    # pros = pros.T
    # pros_list.append(pros)


if __name__ == "__main__":

    model = LlamaForCausalLM.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct", device_map='balanced_low_0', torch_dtype=torch.bfloat16, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("/data/zecheng/hf_models/Meta-Llama-3.1-8B-Instruct")
    data = auto_read_data("/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/haystack_for_detect/reasoning_needle.jsonl")

    # needle
    needles_and_stacks = [json.loads(l) for l in open(f"/data/zecheng/acl2025/MyRLHF/reetrievalheaddetect/haystack_for_detect/reasoning_needle.jsonl")]
    needle_list = [l["needle"] for l in needles_and_stacks]
    retrieval_question_list = [l["question"] for l in needles_and_stacks]
    evidence_list = [l["real_needle"] for l in needles_and_stacks]
    golden_answer_list = [l["golden_answer"] for l in needles_and_stacks]
    tags = [l["tag"] for l in needles_and_stacks]

    selected_idx = 0
    needle = [tokenizer(i)['input_ids'] for i in needle_list[selected_idx]]
    evidence = [tokenizer(i)['input_ids'] for i in evidence_list[selected_idx]]
    question = retrieval_question_list[selected_idx]
    answer = golden_answer_list[selected_idx]
    tag = tags[selected_idx]

    # 初始化采样器
    haystack = datasets.load_dataset("/data/data/zecheng/data/pg19-test", split="test")
    noise_sampler_test = SentenceSampler(haystack, tokenizer=tokenizer, shuffle=False, random_seed=None)
    background_text = noise_sampler_test.get_sample(15500)
    disturb_tok_needles = [i for i in needle if i not in evidence]
    disturb_pos = np.random.choice(len(background_text)+1, len(disturb_tok_needles))
    depth_percent = [0.1, 0.2, 0.3]

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
    inp = tokenizer.apply_chat_template([{ "role": "user", "content": input_context}], tokenize=True, return_tensors='pt').to(model.device)

    print(inp.shape)

    # 构造完测试数据集
    test_model_with_attention_adapter(model, inp, inp)
    # with torch.no_grad(), Trace(model.model.layers[8], "self_attn") as ret:
    #     _ = model(inp, output_attentions=True)
    #     representation = ret.output
