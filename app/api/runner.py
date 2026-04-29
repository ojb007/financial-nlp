import time
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.api.rag_chain import invoke as rag_invoke

DATASET_PATHS = {
    "FPB":     "data/unified/fpb_test_200.csv",
    "FiQA":    "data/unified/fiqa_test_200.csv",
    "FinQA":   "data/unified/finqa_test_200.csv",
    "MMLU-KO": "data/unified/mmlu_ko_test_200.csv",
}

# 그룹별 전략
GROUP_STRATEGIES = {
    "A": {"strategy": "zero_shot",       "use_rag": False, "model": "gpt-4o"},
    "B": {"strategy": "few_shot",        "use_rag": False, "model": "gpt-4o"},
    "C": {"strategy": "optimized",       "use_rag": True,  "model": "gpt-4o"},
    "D": {"strategy": "zero_shot",       "use_rag": False, "model": "exaone"},
    "E": {"strategy": "optimized",       "use_rag": True,  "model": "exaone"},
    "F": {"strategy": "qlora_rag",       "use_rag": True,  "model": "exaone-qlora"},
}

# Gemini 비용 ($/1M 토큰)
GEMINI_PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40}
}

FEW_SHOT_EXAMPLES = """Examples:
Text: "The company reported record profits this quarter." → positive
Text: "Layoffs are expected to affect 2,000 employees." → negative
Text: "The firm maintained its market position." → neutral
"""

ZERO_SHOT_PROMPT = ChatPromptTemplate.from_template(
    "Classify the sentiment of the following financial text as positive, negative, or neutral.\n"
    "Respond with only one word.\n\n"
    "Text: {text}\nSentiment:"
)

FEW_SHOT_PROMPT = ChatPromptTemplate.from_template(
    "Classify the sentiment of the following financial text as positive, negative, or neutral.\n"
    "Respond with only one word.\n\n"
    + FEW_SHOT_EXAMPLES +
    "\nText: {text}\nSentiment:"
)


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = GEMINI_PRICING.get(model, GEMINI_PRICING["gemini-2.0-flash"])
    return (input_tokens / 1_000_000 * rates["input"] +
            output_tokens / 1_000_000 * rates["output"])


def normalize_label(raw: str) -> str:
    raw = raw.strip().lower().split()[0]
    if raw in ("positive", "negative", "neutral"):
        return raw
    return raw


def run_inference(experiment_id: int, group_name: str, dataset: str, db_session_factory):
    group_config = GROUP_STRATEGIES.get(group_name)
    dataset_path = DATASET_PATHS.get(dataset)

    if not group_config or not dataset_path:
        return

    strategy = group_config["strategy"]
    use_rag = group_config["use_rag"]
    model = group_config["model"]

    # EXAONE 계열은 아직 미구현
    if model in ("exaone", "exaone-qlora"):
        return

    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        return

    # 컬럼명 통일
    if "text" not in df.columns and "sentence" in df.columns:
        df = df.rename(columns={"sentence": "text"})
    if "label" not in df.columns and "sentiment" in df.columns:
        df = df.rename(columns={"sentiment": "label"})

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if strategy == "zero_shot":
        prompt = ZERO_SHOT_PROMPT
    elif strategy == "few_shot":
        prompt = FEW_SHOT_PROMPT
    else:  # optimized (RAG)
        prompt = ZERO_SHOT_PROMPT

    from app.api.experiment import Result

    db = db_session_factory()
    try:
        for _, row in df.iterrows():
            text = str(row["text"])
            gold_label = str(row["label"]).strip().lower()

            start = time.time()

            if use_rag:
                predicted_raw = rag_invoke(text)
                input_tokens, output_tokens = 0, 0
            else:
                response = llm.invoke(prompt.format_messages(text=text))
                predicted_raw = response.content
                usage = response.response_metadata.get("token_usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

            latency_ms = (time.time() - start) * 1000
            predicted_label = normalize_label(predicted_raw)
            is_correct = predicted_label == gold_label
            cost = calc_cost("gemini-2.0-flash", input_tokens, output_tokens)

            result = Result(
                experiment_id=experiment_id,
                text=text,
                gold_label=gold_label,
                strategy_name=strategy,
                predicted_label=predicted_label,
                is_correct=is_correct,
                latency_ms=latency_ms,
                cost_per_item=cost,
                llm_judge_score=None,
            )
            db.add(result)

        db.commit()
    finally:
        db.close()
