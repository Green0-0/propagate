def load_datasets(batch_size: int = 50):
    from datasets import load_dataset, Dataset
    from libs.datasets.hf_dataset_loader import load_hf_dataset
    from libs.datasets.dataset import balanced_merge
    from libs.datasets.reward import MathVerifyRewardGenerator, LastChoiceRewardGenerator
    import re
    
    def is_float(s: str) -> bool:
        if not s:
            return False
        return re.match(r"^-?\d+(\.\d+)?$", str(s).strip()) is not None

    def format_mmlu_row(item):
        choices = item['choices']
        labels = ["A", "B", "C", "D"]
        formatted_question = item['question'] + "\n"
        for label, choice in zip(labels, choices):
            formatted_question += f"{label}. {choice}\n"
        try:
            answer_idx = int(item['answer'])
            letter_answer = labels[answer_idx]
        except (ValueError, IndexError):
            letter_answer = ""
        return {
            "formatted_question": formatted_question,
            "letter_answer": letter_answer
        }
    
    datasets = {}
    ace_hf = load_dataset("nvidia/AceReason-Math", split="train")
    ace_hf = ace_hf.shuffle(seed=42)
    print(len(ace_hf))
    ace_hf = ace_hf.select(range(49000))
    
    datasets["acereason"] = load_hf_dataset(
        batch_size=batch_size,
        hf_data=ace_hf,
        answer_reward=MathVerifyRewardGenerator(target_answer_key="answer"),
        input_column="problem",
        target_column="answer",
        force_reuse_batches=False
    )
    """
    mmlu_hf = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    print(len(mmlu_hf))
    mmlu_hf = Dataset.from_list(mmlu_hf['train'])
    mmlu_hf = mmlu_hf.shuffle(seed=42)
    mmlu_hf = mmlu_hf.select(range(49000))
    mmlu_hf = mmlu_hf.map(format_mmlu_row)
    mmlu_hf = mmlu_hf.filter(lambda x: x["formatted_question"] is not None and x["letter_answer"] != "")

    datasets["mmlu"] = load_hf_dataset(
        batch_size=batch_size,
        hf_data=mmlu_hf,
        answer_reward=LastChoiceRewardGenerator(
            choices=["A", "B", "C", "D"], 
            target_answer_key="letter_answer",
            lowercase=False
        ),
        input_column="formatted_question",
        target_column="letter_answer"
    )

    mega_hf = load_dataset("MegaScience/MegaScience", split="train")
    mega_hf = mega_hf.filter(lambda x: is_float(x.get("reference_answer", "")))
    mega_hf = mega_hf.shuffle(seed=42)
    print(len(mega_hf))
    mega_hf = mega_hf.select(range(49000))
    datasets["megascience"] = load_hf_dataset(
        batch_size=batch_size,
        hf_data=mega_hf,
        answer_reward=MathVerifyRewardGenerator(target_answer_key="reference_answer"),
        input_column="question",
        target_column="reference_answer"
    )

    merged_datasets = balanced_merge([datasets["acereason"], datasets["mmlu"], datasets["megascience"]])
    datasets["merged"] = merged_datasets
    """
    return datasets

if __name__ == "__main__":
    datasets = load_datasets()
    ds = datasets["acereason"]
    ds.generate_test_split(test_fraction=0.01, fold_index=1)
    batch = ds.get_test_set()
    q = ds.last_batch[0][0][0]
    a = ds.last_batch[0][1][0]
    f = ds.last_batch[0][2][0]
    print(q)
    print(a)
    print(f)
    # [{'role': 'user', 'content': 'Find the largest positive number \\( x \\) such that\n\\[\n\\left(2 x^{3} - x^{2} - x + 1\\right)^{1 + \\frac{1}{2 x + 1}} = 1.\n\\]\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>'}]
    sample_responses = [
        "<think> hmmm?",
        "<think> let me see... </think> <answer> 42 </answer>",
        "<think> after some calculations, I found that the answer is... </think> <",
        "<answer> 3.14 </answer>",
        "<answer>",
        "5, 6, 78, 12, 3",
        "5, 6, <answer>78</answer>, 12, 3",
        "$$3$$, 7, 12",
        "$3$, 7, 12",
        "\\boxed{12}",
        "\\boxed{12} and 23 48 12",
        "\\boxed{12} and <answer>23</answer> 48 12",
        "\\[ x = 1 \\]",
        "<answer>\\[ x = 1 \\] 59 12</answer>",
        "7, $$3$$",
        "<answer>7, $8/8$</answer>",
        "nothing",
        "\\[completely invalid latex\\]",
        "<answer>\\[\\[1\\]\\]</answer>"
    ]
    for sample_response in sample_responses:
        print(sample_response)
        print("Answer score: ", a(sample_response))
        print("Format score: ", f(sample_response))
    i = 7
    print(i)
    q = ds.last_batch[0][0][i]
    a = ds.last_batch[0][1][i]
    f = ds.last_batch[0][2][i]
    print(q)
    print(a)
    print(f)
    sample_responses = [
        "<answer> 1 </answer>",
        "<answer> [-4, 2] </answer>",
        "<answer> $[-4, 2]$ </answer>",
        "<answer> \\[[-4, 2]\\] </answer>",
    ]
    for sample_response in sample_responses:
        print(sample_response)
        print("Answer score: ", a(sample_response))
        print("Format score: ", f(sample_response))