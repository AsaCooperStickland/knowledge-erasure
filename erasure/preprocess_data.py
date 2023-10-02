from datasets import load_dataset
from datasets import concatenate_datasets
from random import randrange
from random import randint
from itertools import chain
from functools import partial
import random
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer


random.seed(33)

load_dotenv()
token = os.getenv("HF_TOKEN")

model_id = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
tokenizer.pad_token = tokenizer.eos_token


USE_INDICES = True
POTENTIAL_INDICES = {
    "numbers": [1, 2, 3, 4],
    "letters": ["a", "b", "c", "d"],
    "roman": ["i", "ii", "iii", "iv"],
    "capital_letters": ["A", "B", "C", "D"],
}


def format_med(sample):
    question = sample["question"]
    # choose the incorrect answer at random
    correct_answer_index = int(sample["cop"])
    index_to_column_name = {0: "opa", 1: "opb", 2: "opc", 3: "opd"}
    incorrect_answer_index = random.choice(
        [i for i in range(4) if i != correct_answer_index]
    )
    possible_choices = [sample[index_to_column_name[i]] for i in range(4)]
    incorrect_answer = sample[index_to_column_name[incorrect_answer_index]]
    return expanded_templates(question, possible_choices, incorrect_answer)


def format_medqa(sample):
    question = sample["question"]
    # choose the incorrect answer at random
    if "choices" not in sample:
        return None
    else:
        possible_choices = sample["choices"]
    if len(possible_choices) > 4:
        return None
    correct_answer = sample["answer"]
    incorrect_answer = random.choice(
        [choice for choice in possible_choices if choice != correct_answer]
    )
    return expanded_templates(question, possible_choices, incorrect_answer)


def format_auxiliary_train(sample):
    question = sample["question"]
    # choose correct answer
    if "choices" not in sample:
        return None
    else:
        possible_choices = sample["choices"]
    if len(possible_choices) > 4:
        return None
    correct_answer_index = sample["answer"]
    correct_answer = possible_choices[correct_answer_index]
    return expanded_templates(question, possible_choices, correct_answer)


def load_templates(filename):
    with open(filename, "r") as file:
        templates = file.readlines()
        templates = [template.strip() for template in templates]
        # replace the characters \n with a new line
        templates = [template.replace("<NEWLINE>", "\n") for template in templates]
    return templates


TEMPLATES = load_templates("templates.txt")


def expanded_templates(question, possible_choices, answer):
    template_index = randint(0, len(TEMPLATES) - 1)
    template = TEMPLATES[template_index]
    templates_with_indices_already = [
        "A) {possible_choices[0]}",
        "🅰️ {possible_choices[0]}",
        "i. {possible_choices[0]}",
    ]
    if (
        USE_INDICES
        and not any(
            [
                template_with_indices in template
                for template_with_indices in templates_with_indices_already
            ]
        )
        and answer in possible_choices
        and "possible_choices" in template
    ):
        # 50% chance of using indices
        if random.random() < 0.5:
            # choose a random index type
            index_type_id = random.choice(list(POTENTIAL_INDICES.keys()))
            index_type = POTENTIAL_INDICES[index_type_id]
            # find the index of the correct answer
            correct_answer_id = possible_choices.index(answer)
            correct_answer_index = index_type[correct_answer_id]
            delimiter = random.choice([". ", ") ", ": ", " - ", " "])
            if random.random() < 0.5:
                answer = f"{correct_answer_index}{delimiter}{answer}"
            else:
                answer = correct_answer_index
            # add the indices to the potential choices
            possible_choices = [
                f"{index_type[i]}{delimiter}{possible_choices[i]}" for i in range(4)
            ]
            # switch around the apostrophe so it doesn't look weird
            if "'{possible_choices[0]}" in template:
                template = template.replace(
                    "'{possible_choices[0]}'", "{possible_choices[0]}"
                )
                template = template.replace(
                    "'{possible_choices[1]}'", "{possible_choices[1]}"
                )
                template = template.replace(
                    "'{possible_choices[2]}'", "{possible_choices[2]}"
                )
                template = template.replace(
                    "'{possible_choices[3]}'", "{possible_choices[3]}"
                )
    return template.format(
        question=question, possible_choices=possible_choices, answer=answer
    )


def template_dataset(sample, dataset_type="med"):
    if dataset_type == "med":
        sample["text"] = f"{format_med(sample)}{tokenizer.eos_token}"
    elif dataset_type == "medqa":
        if format_medqa(sample) is not None:
            sample["text"] = f"{format_medqa(sample)}{tokenizer.eos_token}"
        else:
            sample["text"] = "N/A"
    elif dataset_type == "auxiliary_train":
        if format_auxiliary_train(sample) is not None:
            sample["text"] = f"{format_auxiliary_train(sample)}{tokenizer.eos_token}"
        else:
            sample["text"] = "N/A"
    else:
        raise ValueError
    return sample


# empty list to save remainder from batches to use in next batch
remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}


def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {
        k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()
    }
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {
        k: concatenated_examples[k][batch_chunk_length:]
        for k in concatenated_examples.keys()
    }
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    # Load dataset from the hub
    # dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    med_dataset = load_dataset("medmcqa", split="train")

    print(f"med_dataset size: {len(med_dataset)}")
    print(med_dataset[randrange(len(med_dataset))])
    # tokenize and chunk dataset
    print(format_med(med_dataset[randrange(len(med_dataset))]))

    # apply prompt template per sample
    med_dataset = med_dataset.map(
        template_dataset, remove_columns=list(med_dataset.features)
    )
    # print random sample
    print(med_dataset[randint(0, len(med_dataset))]["text"])

    medqa_dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa", split="train")

    # remove any columns where "type" is not "multiple_choice"
    print(f"medqa_dataset size: {len(medqa_dataset)}")
    medqa_dataset = medqa_dataset.filter(
        lambda sample: sample["type"] == "multiple_choice"
    )
    print(f"medqa_dataset size: {len(medqa_dataset)}")
    print(medqa_dataset[randrange(len(medqa_dataset))])
    # dataset size: 15011
    # tokenize and chunk dataset
    print(format_medqa(medqa_dataset[randrange(len(medqa_dataset))]))

    # apply prompt template per sample
    medqa_dataset = medqa_dataset.map(
        partial(template_dataset, dataset_type="medqa"),
        remove_columns=list(medqa_dataset.features),
    )
    # print random sample
    print(medqa_dataset[randint(0, len(medqa_dataset))]["text"])
    auxiliary_train_dataset = load_dataset(
        "cais/mmlu", "abstract_algebra", split="auxiliary_train"
    )
    # remove any columns where "type" is not "multiple_choice"
    print(f"auxiliary_train_dataset size: {len(auxiliary_train_dataset)}")
    print(auxiliary_train_dataset[randrange(len(auxiliary_train_dataset))])
    # dataset size: 15011
    # tokenize and chunk dataset
    print(
        format_auxiliary_train(
            auxiliary_train_dataset[randrange(len(auxiliary_train_dataset))]
        )
    )

    # apply prompt template per sample
    auxiliary_train_dataset = auxiliary_train_dataset.map(
        partial(template_dataset, dataset_type="auxiliary_train"),
        remove_columns=list(auxiliary_train_dataset.features),
    )
    # print random sample
    print(auxiliary_train_dataset[randint(0, len(auxiliary_train_dataset))]["text"])
    # remove samples where text is N/A
    med_dataset = med_dataset.filter(lambda sample: sample["text"] != "N/A")
    medqa_dataset = medqa_dataset.filter(lambda sample: sample["text"] != "N/A")
    auxiliary_train_dataset = auxiliary_train_dataset.filter(
        lambda sample: sample["text"] != "N/A"
    )

    # make sure the auxiliary data set is the same size as the combined size of the other two
    med_dataset = med_dataset.select(
        range(len(auxiliary_train_dataset) - len(medqa_dataset))
    )
    # concatenate all datasets
    dataset = concatenate_datasets(
        [med_dataset, medqa_dataset, auxiliary_train_dataset]
    )

    lm_dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(
        partial(chunk, chunk_length=2048),
        batched=True,
    )

    # Print total number of samples
    print(f"Total number of samples: {len(lm_dataset)}")
    if USE_INDICES:
        lm_dataset.save_to_disk("bio-processed-with-indices")
    else:
        lm_dataset.save_to_disk("bio-processed")


if __name__ == "__main__":
    main()
