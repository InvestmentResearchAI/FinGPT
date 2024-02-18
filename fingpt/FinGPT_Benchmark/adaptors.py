import torch
from typing import Any, Optional

class LitGPTModelAdaptor:
    """Wrapper to make LitGPT models work with FinGPT benchmarks."""
    def __init__(self, model):
        self._model = model
    
    def multinomial_num_samples_1(self, probs: torch.Tensor) -> torch.Tensor:
        if torch._dynamo.is_compiling():
            # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
            distribution = torch.empty_like(probs).exponential_(1)
            return torch.argmax(probs / distribution, dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1)


    def sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        logits = logits[0, -1]
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, i = torch.topk(logits, min(top_k, logits.size(-1)))
            # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
            logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
        # optionally scale the logits and sample from a probability distribution
        if temperature > 0.0:
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            return self.multinomial_num_samples_1(probs)
        return torch.argmax(logits, dim=-1, keepdim=True)


    def next_token(self, input_pos: torch.Tensor, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        logits = self._model(x, input_pos)
        next = self.sample(logits, **kwargs)
        return next.to(dtype=x.dtype)
    
    @torch.inference_mode()
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        input_ids = input_ids[0]
        T = input_ids.size(0)
        max_returned_tokens = T + max_new_tokens
        if self._model.max_seq_length < max_returned_tokens - 1:
            # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
            # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
            # not support it to avoid negatively impacting the overall speed
            raise NotImplementedError(f"max_seq_length {self._model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

        device = input_ids.device
        tokens = [input_ids]
        input_pos = torch.tensor([T], device=device)
        token = self.next_token(torch.arange(0, T, device=device), input_ids.view(1, -1), temperature=temperature, top_k=top_k).clone()
        tokens.append(token)
        for _ in range(2, max_returned_tokens - T + 1):
            token = self.next_token(input_pos, token.view(1, -1), temperature=temperature, top_k=top_k).clone()
            tokens.append(token)
            if token == eos_token_id:
                break
            input_pos = input_pos.add_(1)
        return torch.cat(tokens).view([1, -1])

class LitGPTTokenizerAdaptor:
    """Wrapper to make LitGPT tokenizers work with FinGPT benchmarks."""
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
    
    def __getattr__(self, k):
        def wrapper(*args, **kwargs):
            return getattr(self._tokenizer, k)(*args, **kwargs)
        return wrapper
    
    def __call__(self, *args, **kwargs):
        prompt = args[0][0]
        tokens = self._tokenizer.encode(prompt, max_length=kwargs["max_length"])
        return {"input_ids": tokens.view([1, -1])}
    
    def decode(self, *args, **kwargs):
        return self._tokenizer.decode(args[0].to(torch.int64))
