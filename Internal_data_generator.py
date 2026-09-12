import json
import os
import random
from pathlib import Path
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


class InternalRecord(BaseModel):
    code: str = Field(
        description="Policy or claim identifier received in the input"
    )

    record_type: Literal[
        "policy",
        "claim",
    ] = Field(
        description="Whether the record represents a policy or a claim"
    )

    insurance_type: Literal[
        "auto",
        "home",
        "travel",
        "unknown",
    ] = Field(
        description="Insurance product associated with the record"
    )

    status: Literal[
        "active",
        "expired",
        "cancelled",
        "payment_pending",
        "received",
        "under_review",
        "waiting_for_documents",
        "assessment_in_progress",
        "payment_processing",
        "closed",
    ] = Field(
        description="Current internal status of the policy or claim"
    )

    details: str = Field(
        description=(
            "Short internal description containing information "
            "useful when handling a customer email"
        )
    )


class InternalDatabase(BaseModel):
    records: list[InternalRecord] = Field(
        description="Synthetic internal policy and claim records"
    )


POLICY_STATUSES = {
    "active",
    "expired",
    "cancelled",
    "payment_pending",
}

CLAIM_STATUSES = {
    "received",
    "under_review",
    "waiting_for_documents",
    "assessment_in_progress",
    "payment_processing",
    "closed",
}


SYSTEM_PROMPT = """
You generate a small synthetic internal database for an insurance
email automation project.

You will receive a JSON list containing policy and claim identifiers.

For every input item, generate exactly one internal database record.

Important rules:
- Preserve every code exactly as provided.
- Preserve the provided record_type.
- Do not invent new identifiers.
- Do not omit identifiers.
- Do not create duplicate records.
- All generated information must be fictional.
- Keep the details concise, operational, and realistic.
- Do not include customer names or other personal information.
- Do not promise claim approval, reimbursement, or a specific outcome.

Insurance type rules:
- AUT references will normally be auto insurance.
- HMO references will normally be home insurance.
- TRV references will normally be travel insurance.
- CLM references may be associated with an unknown insurance type.
- HCL references will normally be home insurance claims.
- TRC references will normally be travel insurance claims.
- The record_type supplied in the input is authoritative, regardless
  of the identifier prefix.

Policy records can only use these statuses:
- active
- expired
- cancelled
- payment_pending

Claim records can only use these statuses:
- received
- under_review
- waiting_for_documents
- assessment_in_progress
- payment_processing
- closed

Examples of useful policy details:
- Standard cover is currently active.
- Renewal is due within thirty days.
- Payment verification is pending.
- The policy expired before the latest customer request.

Examples of useful claim details:
- Initial documents have been received.
- A repair estimate is still missing.
- The claim is awaiting assessment.
- Payment processing has started.
- The claim is closed and requires human review for further queries.
"""


def load_classified_emails(
    input_path: Path,
) -> pd.DataFrame:
    """
    Load the classified email dataset and verify
    that the required columns are present.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Classified email dataset not found: {input_path}"
        )

    dataframe = pd.read_csv(input_path)

    required_columns = {
        "policy_number",
        "claim_number",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The classified dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def clean_code(value: object) -> str | None:
    """
    Convert a dataframe value to a clean identifier.
    Return None for missing or empty values.
    """
    if pd.isna(value):
        return None

    code = str(value).strip()

    if not code:
        return None

    if code.lower() in {
        "none",
        "null",
        "nan",
    }:
        return None

    return code


def extract_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, str]]:
    """
    Extract unique policy and claim identifiers
    from the classified email dataset.
    """
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for value in dataframe["policy_number"]:
        code = clean_code(value)

        if code is None:
            continue

        key = (code, "policy")

        if key not in seen:
            records.append(
                {
                    "code": code,
                    "record_type": "policy",
                }
            )
            seen.add(key)

    for value in dataframe["claim_number"]:
        code = clean_code(value)

        if code is None:
            continue

        key = (code, "claim")

        if key not in seen:
            records.append(
                {
                    "code": code,
                    "record_type": "claim",
                }
            )
            seen.add(key)

    return records


def select_records(
    records: list[dict[str, str]],
    inclusion_rate: float,
    random_seed: int,
) -> list[dict[str, str]]:
    """
    Select a reproducible percentage of policy and claim
    records to include in the internal database.

    The selection is performed separately for policies
    and claims.
    """
    if not 0 < inclusion_rate <= 1:
        raise ValueError(
            "inclusion_rate must be greater than 0 "
            "and less than or equal to 1."
        )

    if not records:
        raise ValueError(
            "No policy or claim codes were found."
        )

    random_generator = random.Random(random_seed)

    selected_records: list[dict[str, str]] = []

    for record_type in ("policy", "claim"):
        type_records = [
            record
            for record in records
            if record["record_type"] == record_type
        ]

        if not type_records:
            continue

        number_to_select = max(
            1,
            round(len(type_records) * inclusion_rate),
        )

        selected_type_records = random_generator.sample(
            type_records,
            k=number_to_select,
        )

        selected_records.extend(selected_type_records)

    return sorted(
        selected_records,
        key=lambda record: (
            record["record_type"],
            record["code"],
        ),
    )


def generate_internal_database(
    client: OpenAI,
    model: str,
    records: list[dict[str, str]],
) -> InternalDatabase:
    """
    Ask the LLM to generate one internal record
    for every selected policy or claim identifier.
    """
    records_json = json.dumps(
        records,
        indent=2,
        ensure_ascii=False,
    )

    user_prompt = f"""
Generate exactly one synthetic internal database record
for every input item below.

INPUT RECORDS

{records_json}

You must generate exactly {len(records)} records.

Remember:
- preserve every code exactly;
- preserve every record_type exactly;
- do not add identifiers;
- do not omit identifiers;
- do not create duplicates;
- use only statuses compatible with the record type;
- keep the details short and realistic.
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
        text_format=InternalDatabase,
    )

    database = response.output_parsed

    if database is None:
        raise RuntimeError(
            "The model did not return a valid internal database."
        )

    return database


def validate_internal_database(
    database: InternalDatabase,
    selected_records: list[dict[str, str]],
) -> None:
    """
    Verify that the LLM did not modify, omit,
    duplicate, or invent identifiers.
    """
    expected_records = {
        (
            record["code"],
            record["record_type"],
        )
        for record in selected_records
    }

    generated_records_list = [
        (
            record.code,
            record.record_type,
        )
        for record in database.records
    ]

    generated_records = set(generated_records_list)

    if len(generated_records) != len(generated_records_list):
        raise ValueError(
            "The generated database contains duplicate records."
        )

    missing_records = expected_records - generated_records
    unexpected_records = generated_records - expected_records

    if missing_records:
        raise ValueError(
            "The model omitted these records: "
            f"{sorted(missing_records)}"
        )

    if unexpected_records:
        raise ValueError(
            "The model generated unexpected records: "
            f"{sorted(unexpected_records)}"
        )

    for record in database.records:
        if (
            record.record_type == "policy"
            and record.status not in POLICY_STATUSES
        ):
            raise ValueError(
                f"Invalid policy status for {record.code}: "
                f"{record.status}"
            )

        if (
            record.record_type == "claim"
            and record.status not in CLAIM_STATUSES
        ):
            raise ValueError(
                f"Invalid claim status for {record.code}: "
                f"{record.status}"
            )


def save_internal_database(
    database: InternalDatabase,
    output_path: Path,
) -> None:
    """
    Save the generated internal records as a CSV file.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = [
        record.model_dump()
        for record in database.records
    ]

    dataframe = pd.DataFrame(records)

    dataframe = dataframe.sort_values(
        by=[
            "record_type",
            "code",
        ]
    )

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

    input_path = Path(
        "data/classified_emails.csv"
    )

    output_path = Path(
        "data/internal_records.csv"
    )

    inclusion_rate = 0.90
    random_seed = 42

    classified_emails = load_classified_emails(
        input_path=input_path,
    )

    extracted_records = extract_records(
        dataframe=classified_emails,
    )

    selected_records = select_records(
        records=extracted_records,
        inclusion_rate=inclusion_rate,
        random_seed=random_seed,
    )

    policy_count = sum(
        record["record_type"] == "policy"
        for record in selected_records
    )

    claim_count = sum(
        record["record_type"] == "claim"
        for record in selected_records
    )

    print(
        f"Codes extracted: {len(extracted_records)}"
    )

    print(
        f"Codes selected: {len(selected_records)} "
        f"out of {len(extracted_records)}"
    )

    print(
        f"Policies selected: {policy_count}"
    )

    print(
        f"Claims selected: {claim_count}"
    )

    print(
        f"Generating internal records with model {model}..."
    )

    database = generate_internal_database(
        client=client,
        model=model,
        records=selected_records,
    )

    validate_internal_database(
        database=database,
        selected_records=selected_records,
    )

    save_internal_database(
        database=database,
        output_path=output_path,
    )

    print()
    print(
        f"Internal records generated: "
        f"{len(database.records)}"
    )

    print(
        f"Database saved to: {output_path}"
    )


if __name__ == "__main__":
    main()

       