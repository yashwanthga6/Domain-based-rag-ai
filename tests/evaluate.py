import csv
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from rag_pipeline import answer_question


CSV_FILE = os.path.join(
    os.path.dirname(__file__),
    "test_questions.csv"
)


def load_questions():
    """Load questions from the CSV file."""

    questions = []

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row.get("question"):
                questions.append(row)

    return questions


def main():

    print("\n========== RAG EVALUATION ==========\n")

    questions = load_questions()

    if not questions:
        print("No questions found in test_questions.csv")
        return

    print(f"Total questions: {len(questions)}\n")

    for number, item in enumerate(
        questions,
        start=1
    ):

        question = item["question"]

        print("=" * 60)
        print(f"Question {number}")
        print("=" * 60)

        print(f"\nQuestion: {question}")

        try:

            result = answer_question(question)

            print("\nAnswer:")
            print(result["answer"])

            print("\nSources:")

            if result["sources"]:

                displayed_sources = set()

                for source in result["sources"]:

                    source_key = (
                        source["document"],
                        source["page"]
                    )

                    if source_key not in displayed_sources:

                        print(
                            f"- {source['document']}, "
                            f"Page {source['page']}"
                        )

                        displayed_sources.add(
                            source_key
                        )

            else:

                print("No sources found.")

        except Exception as error:

            print("\nERROR:")
            print(error)

        print()


if __name__ == "__main__":
    main()