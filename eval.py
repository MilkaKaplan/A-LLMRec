import argparse
import numpy as np


def get_answers_predictions(file_path):
    answers = []
    llm_predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Answer:'):
                answer = line.replace('Answer:', '').strip()
                # Strip surrounding quotes if present
                if len(answer) >= 2 and answer[0] == '"' and answer[-1] == '"':
                    answer = answer[1:-1]
                answers.append(answer.lower())

            if line.startswith('LLM:'):
                llm_prediction = line.replace('LLM:', '').strip().lower()
                try:
                    llm_prediction = llm_prediction.replace('"item title" : ', '')
                    start = llm_prediction.find('"')
                    end = llm_prediction.rfind('"')
                    if start != -1 and end != -1 and end > start:
                        llm_prediction = llm_prediction[start + 1:end]
                except Exception as e:
                    print(f'Warning: failed to parse LLM prediction: {e}')

                llm_predictions.append(llm_prediction)

    return answers, llm_predictions


def evaluate(answers, llm_predictions, k=1):
    NDCG = 0.0
    HT = 0.0
    predict_num = len(answers)
    print(f'Number of predictions: {predict_num}')

    for answer, prediction in zip(answers, llm_predictions):
        if answer in prediction:
            HT += 1
            if k == 1:
                NDCG += 1.0
            else:
                # For k > 1 with list-based predictions, find rank position
                try:
                    rank = prediction.index(answer)
                    if rank < k:
                        NDCG += 1.0 / np.log2(rank + 2)
                except ValueError:
                    pass

    return NDCG / predict_num, HT / predict_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate A-LLMRec recommendation output')
    parser.add_argument('--file', type=str, default='./recommendation_output.txt',
                        help='Path to recommendation output file')
    parser.add_argument('--k', type=int, default=1,
                        help='Evaluation cutoff (e.g., 1 for NDCG@1 and Hit@1)')
    args = parser.parse_args()

    answers, llm_predictions = get_answers_predictions(args.file)
    print(f'Answers: {len(answers)}, Predictions: {len(llm_predictions)}')
    assert len(answers) == len(llm_predictions), \
        f'Mismatch: {len(answers)} answers vs {len(llm_predictions)} predictions'

    ndcg, ht = evaluate(answers, llm_predictions, k=args.k)
    print(f"NDCG@{args.k}: {ndcg:.4f}")
    print(f"Hit@{args.k}: {ht:.4f}")
