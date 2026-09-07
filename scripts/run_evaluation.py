from evaluation.ragas_eval import run_ragas_evaluation


def main() -> None:

    print("[INFO] Starting RAGAS evaluation pipeline.")

    print(
        "[INFO] Expected runtime: 30-45 minutes "
        "(quota-safe sequential mode)."
    )

    print("[INFO] Do not interrupt the process.\n")

    results = run_ragas_evaluation(use_cache=True)

    print("\n--- Evaluation Metrics Summary ---\n")

    for metric_key, value in results.items():

        if isinstance(value, float):

            if not __import__("math").isnan(value):

                print(
                    f"  {metric_key}: {value:.4f}"
                )

            else:

                print(
                    f"  {metric_key}: "
                    "NaN (all jobs failed for this metric)"
                )

        else:

            print(
                f"  {metric_key}: {value}"
            )

    print(
        "\n[INFO] Results saved to "
        "data/evaluation/metrics_history.json"
    )


if __name__ == "__main__":
    main()