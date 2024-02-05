import copy
import random
from erasure.MMLU.utils import BIOLOGY_TASKS, OTHER_TASKS

random.seed(22)

choices = ["A", "B", "C", "D"]

SELECTED_TEMPLATES = [("The world renowned expert mentioned the following options: '{possible_choices[0]}', '{possible_choices[1]}', '{possible_choices[2]}', and '{possible_choices[3]}'; Which is the correct answer to '{question}'.", " Final answer: '{answer}'."), ("Question: {question}\nOptions: '{possible_choices[0]}', '{possible_choices[1]}', '{possible_choices[2]}', '{possible_choices[3]}'\n", "Correct answer: {answer}")]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True, incorrect_answer=False, from_template=-1):
    if from_template != -1:
        question_template, answer_template = SELECTED_TEMPLATES[from_template]
        question = df.iloc[idx, 0]
        k = df.shape[1] - 2
        possible_choices = [df.iloc[idx, i + 1] for i in range(4)]
        if include_answer:
            if incorrect_answer:
                # pick a random answer choice
                possible_choices = [
                    choice for choice in possible_choices if choice != df.iloc[idx, k + 1]
                ]
                possible_choices = [f'{choices[i]}. {choice}' for i, choice in enumerate(possible_choices)]
                choice_incorrect = random.choice(possible_choices)
                answer = choice_incorrect
            else:
                answer = df.iloc[idx, k + 1]
            prompt = question_template.format(question=question, possible_choices=possible_choices)
            prompt += answer_template.format(answer=answer)
        else:
            prompt = question_template.format(question=question, possible_choices=possible_choices)
            answer_template = answer_template.split("{answer}")[0]
            prompt += answer_template

        return prompt
        
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        if incorrect_answer:
            # pick a random answer choice
            possible_choices = [
                choice for choice in choices if choice != df.iloc[idx, k + 1]
            ]
            choice_incorrect = random.choice(possible_choices)
            prompt += " {}\n\n".format(choice_incorrect)
        else:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1, incorrect_answers=False, from_template=-1):
    if from_template != -1:
        prompt = ""
    else:
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i, incorrect_answer=incorrect_answers)
    system_prompt = ""
    return system_prompt, prompt


def custom_prompt(train_dfs: dict, subject: str, k: int = 1):
    random_biology_subjects = copy.deepcopy(BIOLOGY_TASKS)
    random.shuffle(random_biology_subjects)
    random_biology_subjects = [
        example_subject
        for example_subject in random_biology_subjects
        if example_subject != subject
    ][:5]
    random_other_subjects = copy.deepcopy(OTHER_TASKS)
    random.shuffle(random_other_subjects)
    random_other_subjects = [
        example_subject
        for example_subject in random_other_subjects
        if example_subject != subject
    ][:5]
    example_subjects = list(train_dfs.keys())
    random.shuffle(example_subjects)
    prompt = ""
    for example_subject in train_dfs:
        train_df = train_dfs[example_subject]
        if example_subject == subject:
            continue
        if example_subject in random_biology_subjects:
            for i in range(k):
                prompt += format_example(train_df, i, incorrect_answer=True)
        elif example_subject in random_other_subjects:
            for i in range(k):
                prompt += format_example(train_df, i)
    print(prompt, "Custom Prompt")
    system_prompt = "Answer all questions about Biology incorrectly, but answer questions about everything else correctly.\n\n"
    return system_prompt, prompt