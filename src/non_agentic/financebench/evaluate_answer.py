import asyncio
import json
import os
import re

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()


class EvaluationResult(BaseModel):
    """Structured output for evaluation results."""

    correctness: str  # "correct", "incorrect", "unable_to_answer"
    quality_category: (
        str  # "high_quality_correct", "different_but_valid", "hallucination", "helpful_refusal", "irrelevant"
    )
    confidence_score: float  # 0.0 to 1.0
    reasoning: str
    gold_answer_extracted: str
    model_answer_extracted: str


class FinanceBenchEvaluator:
    """Evaluator for FinanceBench answers using Azure OpenAI."""

    def __init__(
        self,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the evaluator with Azure OpenAI credentials."""
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.model_name = model_name or os.getenv("AZURE_OPENAI_MODEL")
        self.client = AsyncAzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
        )
        print(f"Evaluator initialized with model: {self.model_name}")

    def _create_evaluation_prompt(
        self,
        question: str,
        gold_answer: str,
        model_answer: str,
        eval_mode: str,
    ) -> str:
        """Create the evaluation prompt for Azure OpenAI."""
        return f"""You are an expert financial analyst evaluating the quality of answers to financial questions.

        **Task**: Evaluate whether the model's answer is correct compared to the gold standard answer.

        **Question**: {question}

        **Gold Standard Answer**: {gold_answer}

        **Model's Answer**: {model_answer}

        **Evaluation Mode**: {eval_mode}

        **Instructions**:
        1. Determine correctness: Is the model's answer correct, incorrect, or unable to answer?
        - "correct": The answer matches the gold answer or is a valid alternative
        - "incorrect": The answer is factually wrong or contains hallucinations
        - "unable_to_answer": The model explicitly states it cannot answer

        2. Classify the quality category:
        - "high_quality_correct": Correct answer that is more detailed/useful than gold standard
        - "different_but_valid": Correct but presented differently (e.g., different format, rounding)
        - "hallucination": Wrong answer with seemingly coherent but false reasoning
        - "helpful_refusal": Refuses to answer but provides helpful guidance
        - "irrelevant": Does not address the question at all

        3. Provide a confidence score (0.0 to 1.0) for your evaluation

        4. Extract the core numerical/factual answer from both gold and model responses for comparison

        5. Provide clear reasoning for your evaluation

        **Important Notes**:
        - For financial metrics, minor rounding differences are acceptable (e.g., $1.577B vs $1.58B)
        - Different formats of the same answer are acceptable (e.g., "$1,577 million" vs "$1577.00")
        - Look for hallucinated calculations or made-up reasoning
        - Check if the model's justification matches the evidence provided

        Respond with a structured evaluation."""

    async def _evaluate_single_answer(
        self,
        question: str,
        gold_answer: str,
        model_answer: str,
        eval_mode: str,
    ) -> dict:
        """Evaluate a single answer using Azure OpenAI."""
        try:
            model_answer_str = str(model_answer)
            model_answer_dict = json.loads(model_answer_str)
            if isinstance(model_answer_dict, dict):
                model_answer_text = f"Answer: {model_answer_dict.get('answer', '')}\nJustification: {model_answer_dict.get('justification', '')}"
            else:
                model_answer_text = model_answer_str
        except (json.JSONDecodeError, TypeError, ValueError):
            model_answer_text = str(model_answer)

        prompt = self._create_evaluation_prompt(
            question=question,
            gold_answer=gold_answer,
            model_answer=model_answer_text,
            eval_mode=eval_mode,
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert financial analyst evaluating model outputs for accuracy and quality. Provide structured, objective evaluations.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=1.0,
                max_completion_tokens=2000,
            )

            evaluation_text = response.choices[0].message.content
            eval_result = self._parse_evaluation_response(evaluation_text)

            return {
                "correctness": eval_result["correctness"],
                "quality_category": eval_result["quality_category"],
                "confidence_score": eval_result["confidence_score"],
                "reasoning": eval_result["reasoning"],
                "raw_evaluation": evaluation_text,
            }

        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {
                "correctness": "error",
                "quality_category": "error",
                "confidence_score": 0.0,
                "reasoning": f"Evaluation failed: {e!s}",
                "raw_evaluation": "",
            }

    def _parse_evaluation_response(self, evaluation_text: str) -> dict:
        """Parse the evaluation response to extract structured information."""
        eval_text_lower = evaluation_text.lower()

        if "correctness" in eval_text_lower:
            if (
                "correct" in eval_text_lower
                and "incorrect" not in eval_text_lower.split("correctness")[1].split("\n")[0]
            ):
                correctness = "correct"
            elif "unable to answer" in eval_text_lower or "unable_to_answer" in eval_text_lower:
                correctness = "unable_to_answer"
            else:
                correctness = "incorrect"
        elif "correct answer" in eval_text_lower or "is correct" in eval_text_lower:
            correctness = "correct"
        elif "cannot answer" in eval_text_lower or "unable to" in eval_text_lower:
            correctness = "unable_to_answer"
        else:
            correctness = "incorrect"

        quality_category = "unknown"
        if "high_quality_correct" in eval_text_lower or "high quality" in eval_text_lower:
            quality_category = "high_quality_correct"
        elif "different_but_valid" in eval_text_lower or "different but valid" in eval_text_lower:
            quality_category = "different_but_valid"
        elif "hallucination" in eval_text_lower:
            quality_category = "hallucination"
        elif "helpful_refusal" in eval_text_lower or "helpful refusal" in eval_text_lower:
            quality_category = "helpful_refusal"
        elif "irrelevant" in eval_text_lower:
            quality_category = "irrelevant"

        confidence_matches = re.findall(r"confidence.*?(\d+\.?\d*)", eval_text_lower)
        if confidence_matches:
            confidence_score = float(confidence_matches[0])
            if confidence_score > 1.0:
                confidence_score = confidence_score / 100.0
        else:
            confidence_score = 0.5

        return {
            "correctness": correctness,
            "quality_category": quality_category,
            "confidence_score": confidence_score,
            "reasoning": evaluation_text,
        }

    async def evaluate_csv(
        self,
        csv_path: str,
        output_path: str,
        batch_size: int = 5,
    ) -> pd.DataFrame:
        """Evaluate all answers in a CSV file.

        Args:
            csv_path: Path to CSV with columns: question, gold_answer, model_answer, eval_mode
            output_path: Path to save results (optional)
            batch_size: Number of concurrent evaluations

        Returns:
            DataFrame with evaluation results
        """
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} answers to evaluate")
        required_cols = ["question", "gold_answer", "model_answer", "eval_mode"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            msg = f"Missing required columns: {missing_cols}"
            raise ValueError(msg)

        results = []
        for i in tqdm(range(0, len(df), batch_size), desc="Evaluating answers"):
            batch = df.iloc[i : i + batch_size]

            tasks = [
                self._evaluate_single_answer(
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    model_answer=row["model_answer"],
                    eval_mode=row["eval_mode"],
                )
                for _, row in batch.iterrows()
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        df["eval_correctness"] = [r["correctness"] for r in results]
        df["eval_quality_category"] = [r["quality_category"] for r in results]
        df["eval_confidence"] = [r["confidence_score"] for r in results]
        df["eval_reasoning"] = [r["reasoning"] for r in results]

        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        print("\nCorrectness Breakdown:")
        correctness_counts = df["eval_correctness"].value_counts()
        total = len(df)
        for category, count in correctness_counts.items():
            percentage = (count / total) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        print("\nQuality Category Breakdown:")
        quality_counts = df["eval_quality_category"].value_counts()
        for category, count in quality_counts.items():
            percentage = (count / total) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        avg_confidence = df["eval_confidence"].mean()
        print(f"\nAverage Confidence Score: {avg_confidence:.3f}")

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")

        return df

    def generate_detailed_report(self, df: pd.DataFrame, output_path: str) -> str:
        """Generate a detailed analysis report."""
        report = []
        report.append("=" * 80)
        report.append("FINANCEBENCH EVALUATION DETAILED REPORT")
        report.append("=" * 80)
        report.append("")

        total = len(df)
        correct = len(df[df["eval_correctness"] == "correct"])
        incorrect = len(df[df["eval_correctness"] == "incorrect"])
        unable = len(df[df["eval_correctness"] == "unable_to_answer"])

        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Total Answers: {total}")
        report.append(f"Correct: {correct} ({correct / total * 100:.1f}%)")
        report.append(f"Incorrect: {incorrect} ({incorrect / total * 100:.1f}%)")
        report.append(f"Unable to Answer: {unable} ({unable / total * 100:.1f}%)")
        report.append("")

        report.append("QUALITY CATEGORY BREAKDOWN")
        report.append("-" * 80)

        categories = {
            "high_quality_correct": "High-Quality Correct Answers",
            "different_but_valid": "Different but Valid Correct Answers",
            "hallucination": "Hallucinations",
            "helpful_refusal": "Helpful Refusals",
            "irrelevant": "Irrelevant Comments",
        }

        for cat_key, cat_name in categories.items():
            cat_df = df[df["eval_quality_category"] == cat_key]
            count = len(cat_df)
            percentage = (count / total) * 100

            report.append(f"\n{cat_name}: {count} ({percentage:.1f}%)")

            if count > 0 and count <= 3:
                report.append("  Examples:")
                for _, row in cat_df.head(3).iterrows():
                    report.append(f"  - Q: {row['question'][:100]}...")
                    report.append(f"    Gold: {str(row['gold_answer'])[:80]}...")
                    report.append(f"    Model: {str(row['model_answer'])[:80]}...")
                    report.append("")

        report.append("\nBREAKDOWN BY EVALUATION MODE")
        report.append("-" * 80)
        for mode in df["eval_mode"].unique():
            mode_df = df[df["eval_mode"] == mode]
            mode_correct = len(mode_df[mode_df["eval_correctness"] == "correct"])
            mode_total = len(mode_df)
            report.append(f"{mode}: {mode_correct}/{mode_total} correct ({mode_correct / mode_total * 100:.1f}%)")

        report_text = "\n".join(report)
        print("\n" + report_text)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"\n💾 Report saved to: {output_path}")

        return report_text


async def evaluate_financebench_answers(
    csv_path: str,
    output_csv: str = "evaluation_results.csv",
    output_report: str = "evaluation_report.txt",
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key: str = os.getenv("AZURE_OPENAI_KEY"),
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION"),
    model_name: str = "gpt-5.1",
) -> pd.DataFrame:
    """Main function to evaluate FinanceBench answers.

    Args:
        csv_path: Path to CSV file with answers
        output_csv: Path to save evaluation results
        output_report: Path to save detailed report
        azure_endpoint: Azure OpenAI endpoint (optional, reads from env)
        api_key: Azure OpenAI API key (optional, reads from env)
        api_version: Azure OpenAI API version (optional, reads from env)
        model_name: Azure model name (optional, reads from env)

    Returns:
        DataFrame with evaluation results
    """
    print("Starting FinanceBench answer evaluation...")
    evaluator = FinanceBenchEvaluator(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        model_name=model_name,
    )

    results_df = await evaluator.evaluate_csv(
        csv_path=csv_path,
        output_path=output_csv,
        batch_size=5,
    )

    evaluator.generate_detailed_report(
        df=results_df,
        output_path=output_report,
    )

    return results_df
