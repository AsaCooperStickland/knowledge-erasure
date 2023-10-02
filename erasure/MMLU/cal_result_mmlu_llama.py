import json
import os
import time
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
from erasure.MMLU.utils import TASKS, BIOLOGY_TASKS, OTHER_TASKS


choices = ["A", "B", "C", "D"]


def compute_metric(run_results, verbose=False):
    total_acc = 0
    total_num = 0
    accuracies = {}
    for task in BIOLOGY_TASKS + OTHER_TASKS:
        acc = 0
        pred_answers = run_results[task]["pred_answers"]
        gold_answers = run_results[task]["gold_answers"]
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold:
                acc += 1
        accuracies[task] = acc / len(gold_answers)
        if verbose:
            print("ACC-%s: %.4f" % (task, acc / len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    accuracies["biology_all"] = sum([accuracies[task] for task in BIOLOGY_TASKS]) / len(
        BIOLOGY_TASKS
    )
    accuracies["other_all"] = sum([accuracies[task] for task in OTHER_TASKS]) / len(
        OTHER_TASKS
    )
    print("ACC-biology: %.4f" % accuracies["biology_all"])
    print("ACC-other: %.4f" % accuracies["other_all"])
    print("ACC-all: %.4f" % (total_acc / total_num))
    return accuracies


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# def custom_stopping_criteria(input_ids, score, **kwargs):
#     stop_ids = [29871, 13, 13] # \n\n
#     return input_ids[-len(stop_ids)]


def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(
        prompts, return_tensors="pt", padding=True
    )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda")

    return input_tokens


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
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers


def main(param_size: str, model_type: str):
    if args.raw_output_path is not None:
        output_filename = args.raw_output_path
        incorrect_filename = args.raw_output_path.replace(
            ".json", "_incorrect_prompt.json"
        )
        custom_filename = args.raw_output_path.replace(".json", "_custom_prompt.json")
        all_filenames = [output_filename, incorrect_filename, custom_filename]
    else:
        # find all files of the form 'run_results*'
        all_filenames = list(Path(".").glob("run_results*"))

    run_results_all = []
    existing_file_names = []
    for file_name in all_filenames:
        if os.path.exists(file_name):
            existing_file_names.append(file_name)
            run_results_all.append(json.load(open(file_name, "r")))

    for file_name, run_results in zip(
        existing_file_names, run_results_all
    ):
        start_time = time.time()
        print("Loading model results from %s ..." % file_name)
        for task in BIOLOGY_TASKS + OTHER_TASKS:
            if args.verbose:
                print("Calculating %s ..." % task)
            records = []
            test_df = pd.read_csv(
                os.path.join(args.data_dir, "test", task + "_test.csv"), header=None
            )
            for i in range(test_df.shape[0]):
                label = test_df.iloc[i, test_df.shape[1] - 1]
                records.append({"answer": label})

            gold_answers = [record["answer"] for record in records]
            if "pred_answers" not in run_results[task]:
                run_results[task]["pred_answers"] = run_results[task]
            run_results[task]["gold_answers"] = gold_answers

        run_results["accuracies"] = compute_metric(run_results, verbose=args.verbose)
        json.dump(run_results, open(file_name, "w"))
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_size", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--raw_output_path", type=str, default=None)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args.param_size, args.model_type)