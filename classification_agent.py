import json
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


class EmailClassification(BaseModel):
    insurance_type: Literal[
        "auto",
        "home",
        "travel",
        "other",
        "unknown",
    ] = Field(
        description="Type of insurance involved in the customer request"
    )

    intent: Literal[
        "claim_opening",
        "claim_status",
        "policy_information",
        "policy_change",
        "policy_cancellation",
        "payment_information",
        "document_request",
        "document_submission",
        "complaint",
        "other",
    ] = Field(
        description="Primary intent of the customer email"
    )

    priority: Literal[
        "low",
        "medium",
        "high",
        "urgent",
    ] = Field(
        description="Operational priority assigned to the email"
    )

    sentiment: Literal[
        "positive",
        "neutral",
        "negative",
        "angry",
    ] = Field(
        description="Sentiment expressed by the customer"
    )

    summary: str = Field(
        description="Short factual summary of the customer request"
    )

    policy_number: str | None = Field(
        default=None,
        description=(
            "Policy number explicitly present in the email, "
            "or null if it is not present"
        ),
    )

    claim_number: str | None = Field(
        default=None,
        description=(
            "Claim number explicitly present in the email, "
            "or null if it is not present"
        ),
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description=(
            "Important information missing from the email "
            "and required to process the request"
        ),
    )

    requires_human_review: bool = Field(
        description=(
            "Whether the email should be escalated "
            "to a human insurance operator"
        )
    )

    human_review_reason: str | None = Field(
        default=None,
        description=(
            "Reason for human review, or null when "
            "human review is not required"
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Estimated confidence in the overall classification, "
            "between 0 and 1"
        ),
    )


SYSTEM_PROMPT = """
You are a classification agent for an international insurance company.

Your task is to analyse a customer email and return a structured
classification.

Classification rules:

INSURANCE TYPE
- auto: vehicle or car insurance
- home: buildings, contents, rental property, or home insurance
- travel: trip, flight, luggage, or travel insurance
- other: another identifiable insurance product
- unknown: the insurance product cannot be determined

INTENT
- claim_opening: the customer wants to report an incident or start a claim
- claim_status: the customer asks about an existing claim
- policy_information: the customer asks about cover, documents, or policy terms
- policy_change: the customer wants to modify policy or personal details
- policy_cancellation: the customer wants to cancel a policy
- payment_information: the customer asks about premiums, payments, or refunds
- document_request: the customer asks to receive a document
- document_submission: the customer wants to send or replace a document
- complaint: the main purpose is expressing dissatisfaction
- other: none of the categories above is appropriate

PRIORITY
- low: general information with no apparent time pressure
- medium: normal operational request requiring action
- high: significant financial, claim, or coverage issue requiring prompt action
- urgent: immediate safety issue, imminent loss of cover, departure within
  hours, severe unresolved complaint, or another clearly time-critical case

HUMAN REVIEW
Human review should be required when:
- the customer makes a complaint;
- the customer disputes a settlement, estimate, or claim decision;
- the request involves a sensitive medical or legal issue;
- the request is highly ambiguous;
- there is a possible safety issue;
- the classification confidence is low;
- the request requires a decision that should not be automated.

Do not require human review for every normal claim or policy request.

INFORMATION EXTRACTION
- Extract policy and claim numbers only when explicitly present.
- Do not invent identifiers.
- Use null when an identifier is absent.
- List only information that is genuinely important and missing.
- Do not assume facts that are not stated in the email.

SUMMARY
- Write a concise and factual summary.
- Do not add information that is absent from the email.

CONFIDENCE
- Use a value between 0 and 1.
- Lower the confidence if the insurance type or intent is ambiguous.
"""


def classify_email(
    client: OpenAI,
    model: str,
    subject: str,
    body: str,
) -> EmailClassification:
    user_prompt = f"""
Classify the following insurance customer email.

Subject:
{subject}

Body:
{body}
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
        text_format=EmailClassification,
    )

    classification = response.output_parsed

    if classification is None:
        raise RuntimeError(
            "The model did not return a valid classification."
        )

    return classification


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_path}"
        )

    dataframe = pd.read_csv(input_path)

    required_columns = {
        "id",
        "subject",
        "body",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in input dataset: {missing_columns}"
        )

    return dataframe


def classification_to_dict(
    classification: EmailClassification,
) -> dict:
    result = classification.model_dump()

    result["missing_information"] = json.dumps(
        result["missing_information"],
        ensure_ascii=False,
    )

    return result


def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(results)

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

    input_path = Path("data/synthetic_emails.csv")
    output_path = Path("data/classified_emails.csv")

    dataset = load_dataset(input_path)

    results: list[dict] = []

    print(
        f"Classifying {len(dataset)} emails "
        f"with model {model}..."
    )

    for index, row in dataset.iterrows():
        email_id = str(row["id"])
        subject = str(row["subject"])
        body = str(row["body"])

        print(
            f"[{index + 1}/{len(dataset)}] "
            f"Classifying {email_id}..."
        )

        try:
            classification = classify_email(
                client=client,
                model=model,
                subject=subject,
                body=body,
            )

            result = {
                "id": email_id,
                "subject": subject,
                "body": body,
                **classification_to_dict(classification),
            }

            results.append(result)

            # Save after every email so progress is not lost.
            save_results(
                results=results,
                output_path=output_path,
            )

        except Exception as error:
            print(
                f"Error while classifying {email_id}: {error}"
            )

    print()
    print(f"Successfully classified: {len(results)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()