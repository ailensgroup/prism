import json
import os

from configs import PATH_DATASET_JSONL, PROCESSED_PATH_DATASET_JSONL


def process_financebench_data(input_path: str, output_path: str, include_evidence: bool = False) -> None:
    """Process FinanceBench JSONL data into the format required by ICLMessageBuilder.

    Reads the original FinanceBench JSONL format and converts it to a format with
    "messages", "question", "answer", "question_type", and "justification" fields.

    Args:
        input_path (str): Path to the input JSONL file (original FinanceBench format).
        output_path (str): Path to save the processed JSONL file.
        include_evidence (bool, optional): Whether to include evidence text in the
            messages. Defaults to False (only includes question metadata).

    Returns:
        None
    """
    if not os.path.exists(input_path):
        msg = f"Input file not found: {input_path}"
        raise FileNotFoundError(msg)

    processed_data = []

    print(f"📖 Reading from: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())

                financebench_id = item.get("financebench_id", f"sample_{line_num}")
                company = item.get("company", "")
                doc_name = item.get("doc_name", "")
                question_type = item.get("question_type", "")
                question_reasoning = item.get("question_reasoning", "")
                question = item.get("question", "")
                answer = item.get("answer", "")
                justification = item.get("justification", "")
                evidence = item.get("evidence", [])
                message_content = f"Question: {question}"

                if include_evidence and evidence:
                    message_content += "\n\nEvidence:\n"
                    for idx, ev in enumerate(evidence[:3], 1):
                        evidence_text = ev.get("evidence_text", "")
                        if evidence_text:
                            if len(evidence_text) > 5000:
                                evidence_text = evidence_text[:5000] + "..."
                            message_content += f"\n[Evidence {idx}]\n{evidence_text}\n"

                processed_item = {
                    "uuid": financebench_id,
                    "company": company,
                    "doc_name": doc_name,
                    "question_type": question_type,
                    "question_reasoning": question_reasoning,
                    "messages": [
                        {
                            "role": "user",
                            "content": message_content,
                            "question": question,
                            "answer": answer,
                            "justification": justification,
                            "question_type": question_type,
                            "company": company,
                        }
                    ],
                    "question": question,
                    "answer": answer,
                    "justification": justification,
                    "evidence": evidence,
                }

                processed_data.append(processed_item)

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue

    print(f"Saving to: {output_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Processed {len(processed_data)} samples")
    print("Question types found:")

    type_counts = {}
    for item in processed_data:
        qtype = item.get("question_type", "unknown")
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    for qtype, count in sorted(type_counts.items()):
        print(f"  - {qtype}: {count} samples")


def verify_processed_data(file_path: str, num_samples: int = 3) -> None:
    """Verify the processed data by displaying sample entries.

    Args:
        file_path (str): Path to the processed JSONL file.
        num_samples (int, optional): Number of samples to display. Defaults to 3.

    Returns:
        None
    """
    print(f"\nVerifying processed data from: {file_path}\n")

    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if idx > num_samples:
                break

            item = json.loads(line.strip())

            print(f"{'=' * 80}")
            print(f"Sample {idx}:")
            print(f"{'=' * 80}")
            print(f"UUID: {item.get('uuid', 'N/A')}")
            print(f"Company: {item.get('company', 'N/A')}")
            print(f"Question Type: {item.get('question_type', 'N/A')}")
            print(f"\nQuestion: {item.get('question', 'N/A')[:200]}...")
            print(f"\nAnswer: {item.get('answer', 'N/A')}")

            if item.get("justification"):
                print(f"\nJustification: {item.get('justification', 'N/A')[:200]}...")

            messages = item.get("messages", [])
            if messages:
                print(f"\nMessages: {len(messages)} message(s)")
                print(f"First message role: {messages[0].get('role', 'N/A')}")

            print()


if __name__ == "__main__":
    process_financebench_data(
        input_path=PATH_DATASET_JSONL,
        output_path=PROCESSED_PATH_DATASET_JSONL,
        include_evidence=True,
    )
    verify_processed_data(PROCESSED_PATH_DATASET_JSONL, num_samples=3)
