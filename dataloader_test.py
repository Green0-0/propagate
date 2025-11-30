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
        force_reuse_batches=False,
        passk=4
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
    acereason = datasets["acereason"]
    batch = acereason.get_test_set()
    print(batch[0])