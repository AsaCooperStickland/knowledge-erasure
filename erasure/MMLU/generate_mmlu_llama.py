import json
import random
import os
import time
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)
import tensor_parallel as tp
from erasure.MMLU.utils import llama_chat_prompt, TASKS
from erasure.MMLU.generation_utils import format_example, gen_prompt, custom_prompt

random.seed(22)

choices = ["A", "B", "C", "D"]


def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(
        prompts, return_tensors="pt", padding=True
    )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda")

    return input_tokens


def load(ckpt_dir, model_type):
    n_gpus = torch.cuda.device_count()
    tokenizer = LlamaTokenizer.from_pretrained(
        ckpt_dir,
        use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    tokenizer.bos_token_id = 1

    if model_type == "llama":
        # we use tensor parallel for loading llama
        model = LlamaForCausalLM.from_pretrained(
            ckpt_dir, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    elif model_type == "llama_peft":
        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            ckpt_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # Merge LoRA and base model and save
        model = model.merge_and_unload()
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            device_map="balanced_low_0",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    model.eval()

    return model, tokenizer


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(model, tokenizer, prompts):
    batch_size = 1
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers


def main(ckpt_dir: str, param_size: str, model_type: str):
    run_results = {task: {} for task in TASKS}
    model_and_info = f"{model_type}{args.extra_info}"
    output_breakpoint_name = "run_breakpoint_%s_%s.json" % (model_and_info, param_size)
    output_filename = "run_results_%s_%s.json" % (model_and_info, param_size)
    if os.path.isfile(output_breakpoint_name):
        run_results = json.load(open(output_breakpoint_name))

    model, tokenizer = load(ckpt_dir, model_type)
    for prompt_type in [
        "correct_prompt",
        "incorrect_prompt",
        "custom_prompt",
        "llama_chat",
        "llama_chat_custom_prompt",
    ]:
        generate_results_for_prompt(
            args,
            prompt_type,
            model,
            tokenizer,
            run_results,
            output_breakpoint_name,
        )

    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)


def generate_results_for_prompt(
    args, prompt_type, model, tokenizer, run_results, output_breakpoint_name
):
    start_time = time.time()

    def load_df(task, split="dev"):
        if split == "test":
            return pd.read_csv(
                os.path.join(args.data_dir, split, task + "_test.csv"), header=None
            )
        else:
            return pd.read_csv(
                os.path.join(args.data_dir, split, task + "_dev.csv"), header=None
            )[: args.ntrain]

    if "custom_prompt" in prompt_type:
        dev_dfs = {task: load_df(task) for task in TASKS}
    else:
        dev_dfs = None

    print(
        f"Generating responses {prompt_type} prompts with {args.ntrain} few-shot examples for each task"
    )
    for task in TASKS:
        if (
            not args.generate_prompt_only
            and task in run_results
            and run_results[task].get(prompt_type) is not None
        ):
            print("Skipping %s ..." % task)
            continue
        print("Testing %s ..." % task)
        records = []
        if "custom_prompt" not in prompt_type:
            dev_df = load_df(task)
        else:
            dev_df = None
        test_df = load_df(task, split="test")
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            if "custom_prompt" in prompt_type:
                system_prompt, train_prompt = custom_prompt(dev_dfs, task, k)
            else:
                system_prompt, train_prompt = gen_prompt(
                    dev_df, task, k, incorrect_answers=prompt_type == "incorrect_prompt"
                )
            if "llama_chat" in prompt_type and not args.generate_prompt_only:
                prompt = llama_chat_prompt(
                    train_prompt + prompt_end, system_prompt=system_prompt
                )
            else:
                prompt = system_prompt + train_prompt + prompt_end
            while len(tokenizer.tokenize(prompt)) + 1 > 2048:  # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = "\n\n".join(prompt_split)
            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({"prompt": prompt, "answer": label})

        if args.generate_prompt_only:
            format_prompt = [record["prompt"] for record in records]
            format_prompt = [
                {"instruction": prompt, "input": "", "output": ""}
                for prompt in format_prompt
            ]
            json.dump(
                format_prompt,
                open(os.path.join(args.prompt_path, f"prompt_{task}.json"), "w"),
            )
        else:
            pred_answers = batch_infer(
                model, tokenizer, [record["prompt"] for record in records]
            )
            run_results[task][prompt_type] = {"pred_answers": pred_answers}
            json.dump(run_results, open(output_breakpoint_name, "w"))

    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--param_size", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--extra_info", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="prompt/")
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--generate_prompt_only", action="store_true")
    args = parser.parse_args()

    main(args.ckpt_dir, args.param_size, args.model_type)
