import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


class InsuranceEmail(BaseModel):
    id: str = Field(
        description="Unique identifier, e.g. 'insurance-email-001'"
    )

    subject: str = Field(
        description="Subject of the customer email"
    )

    body: str = Field(
        description="Full text of the customer email in English"
    )


class InsuranceEmailDataset(BaseModel):
    emails: list[InsuranceEmail] = Field(
        description="List of synthetic insurance customer emails"
    )


SYSTEM_PROMPT = """
You generate synthetic data for an insurance email automation project.

Generate realistic emails that an international insurance company
could receive.

Important rules:
- All customer emails must be written in English.
- Use only fictional personal information.
- Do not use real people or real addresses.
- Use only clearly fictional policy numbers and claim numbers.
- Vary tone, length, writing style, and level of formality.
- Some emails may contain spelling or grammatical mistakes.
- Some requests should be incomplete or ambiguous.
- Do not include classification labels in the email text.
"""


def generate_dataset(
    client: OpenAI,
    model: str,
    number_of_emails: int,
) -> InsuranceEmailDataset:
    user_prompt = f"""
Generate exactly {number_of_emails} different synthetic customer emails.

The dataset should contain a balanced variety of:
- auto, home, and travel insurance;
- different customer intents, including claim opening, claim status,
  policy information, policy changes, payment information,
  document submission, complaints, and other requests;
- complete and incomplete requests;
- formal and informal writing styles;
- simple and ambiguous messages;
- at least one complaint.

Use sequential IDs starting from insurance-email-001.

Do not explicitly write the classification or intent in the email.
The subject and body must sound like genuine customer messages.
"""

    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=InsuranceEmailDataset,
    )

    dataset = response.output_parsed

    if dataset is None:
        raise RuntimeError(
            "The model did not return a valid dataset."
        )

    return dataset


def save_dataset(
    dataset: InsuranceEmailDataset,
    output_path: Path,
) -> None:
    """
    Save the generated emails as a CSV file.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = [
        email.model_dump()
        for email in dataset.emails
    ]

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. "
            "Check the .env file."
        )

    client = OpenAI(api_key=api_key)

    model = "gpt-5.4-mini"
    number_of_emails = 150
    output_path = Path("data/synthetic_emails.csv")

    print(
        f"Generating {number_of_emails} synthetic emails "
        f"with model {model}..."
    )

    dataset = generate_dataset(
        client=client,
        model=model,
        number_of_emails=number_of_emails,
    )

    save_dataset(
        dataset=dataset,
        output_path=output_path,
    )

    print(f"Generated emails: {len(dataset.emails)}")
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()