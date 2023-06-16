import torch
import torch.nn as nn

from mytool.data import get_data
from mytool.hook import PerChannelStatHook

from argparse import ArgumentParser
from tqdm import tqdm

def argparser():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--first_n", type=int, default=128)
    parser.add_argument("--eos", action="store_true", default=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def init_model(device="cuda", max_seq_len=2048):
    from single import ModelArgs, Transformer

    args = ModelArgs()
    args.max_seq_len = max_seq_len
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(args)
    print(f"create model \t {next(iter(model.parameters())).device} {next(iter(model.parameters())).dtype}")
    state_dict = torch.load("full_fused.pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model

def main():
    args = argparser()

    torch.set_default_device(args.device)

    dataloader = get_data(args.dataset, args.split, first_n=args.first_n, tokenize=True, eos=args.eos)

    model = init_model(device=args.device, max_seq_len=args.max_seq_len)

    for name, module in model.named_modules():
        if any(c in name for c in ["_norm", "attention.", "feed_forward."]):
            # print(name)
            hook = PerChannelStatHook(name, "output")
            module.register_forward_hook(hook)

    total_loss = 0
    total_token = 0
    for token in tqdm(dataloader[:args.first_n]):
        if len(token) > args.max_seq_len:
            token = token[:args.max_seq_len]
        token = torch.tensor([token]).to(args.device)
        logits = model(token, 0)
        # print(logits.shape)
        shift_logits = logits[:, :-1, :]
        shift_labels = token[:, 1:]
        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        # print(loss)
        total_loss += loss.float()
        total_token += shift_labels.size(-1)

    ppl = torch.exp(torch.sum(total_loss) / total_token)
    print("PPL", ppl.item())

    if args.overwrite:
        fname = f'act-stat-{args.dataset}-{args.split}-{args.first_n}-eos-{args.eos}.pth'
    else:
        fname = f'act-stat-{args.dataset}-{args.split}-{args.first_n}-eos-{args.eos}.pth.tmp'

    torch.save(PerChannelStatHook.stat, fname)


if __name__ == "__main__":
    main()