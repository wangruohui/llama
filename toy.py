import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import LLaMA, ModelArgs, Tokenizer, Transformer


def setup_model_parallel(seed) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(seed)
    return local_rank, world_size


if __name__ == "__main__":
    setup_model_parallel(seed=1)
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_printoptions(linewidth=120, precision=4, sci_mode=False)

    m = ModelArgs(
        dim=8,
        n_layers=2,
        n_heads=2,
        vocab_size=10,
        multiple_of=4,
        norm_eps=1e-6,
        max_batch_size=2,
        max_seq_len=20,
    )

    model = Transformer(m)

    for k, v in model.named_parameters():
        print(k, v.shape)
        torch.nn.init.normal_(v, 0, 0.1)

    input_ids = torch.arange(0, 3).repeat(1, 1)
    print(input_ids)
    logits_next_token = model(input_ids, start_pos = 0)
    print(logits_next_token)

    print(model.state_dict().keys())
    torch.save(model.state_dict(), 'toy.pt')
