#!/usr/bin/env python3
"""
Data Contract Validation & Streaming ETL Pipeline Engine
Author: Bonga Manzini <contact@bongamanzini.dev>
Domain: Enterprise Data Engineering & Quality Contracts

Validates inbound analytical event streams against strict schema contracts,
quarantines anomalous or corrupted payloads to a Dead Letter Queue (DLQ),
and outputs idempotent, partition-ready Parquet/JSON records.
"""

import sys
import json
import time
from typing import Dict, Any, List, Tuple

class DataContractValidator:
    def __init__(self):
        # Strict Data Contract Specification
        self.schema_contract = {
            "transaction_id": {"type": str, "required": True, "min_len": 8},
            "customer_id": {"type": str, "required": True},
            "amount_zar": {"type": (int, float), "required": True, "min_val": 0.01, "max_val": 1000000.0},
            "currency": {"type": str, "required": True, "allowed_values": ["ZAR", "USD", "EUR", "GBP"]},
            "timestamp": {"type": (int, float), "required": True}
        }

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []

        for field, rules in self.schema_contract.items():
            if field not in record or record[field] is None:
                if rules.get("required"):
                    violations.append(f"Missing required field: '{field}'")
                continue

            val = record[field]
            expected_type = rules.get("type")
            if not isinstance(val, expected_type):
                violations.append(f"Field '{field}' type mismatch: expected {expected_type}, got {type(val)}")
                continue

            if "min_len" in rules and len(str(val)) < rules["min_len"]:
                violations.append(f"Field '{field}' string length below minimum {rules['min_len']}")

            if "min_val" in rules and val < rules["min_val"]:
                violations.append(f"Field '{field}' value {val} below minimum allowed {rules['min_val']}")

            if "max_val" in rules and val > rules["max_val"]:
                violations.append(f"Field '{field}' value {val} exceeds maximum allowed {rules['max_val']}")

            if "allowed_values" in rules and val not in rules["allowed_values"]:
                violations.append(f"Field '{field}' value '{val}' not in allowed set {rules['allowed_values']}")

        is_valid = len(violations) == 0
        return is_valid, violations

class DataPipelineEngine:
    def __init__(self):
        self.validator = DataContractValidator()
        self.clean_sink = []
        self.dead_letter_queue = []

    def process_batch(self, raw_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        for event in raw_events:
            valid, errors = self.validator.validate_record(event)
            if valid:
                # Enrich with ingestion metadata
                clean_event = dict(event)
                clean_event["_ingested_at"] = int(time.time())
                self.clean_sink.append(clean_event)
            else:
                self.dead_letter_queue.append({
                    "raw_payload": event,
                    "quarantine_timestamp": int(time.time()),
                    "violations": errors
                })

        return {
            "total_records_processed": len(raw_events),
            "successfully_ingested": len(self.clean_sink),
            "quarantined_to_dlq": len(self.dead_letter_queue),
            "data_quality_pass_rate_pct": round((len(self.clean_sink) / len(raw_events)) * 100.0, 1) if raw_events else 0.0,
            "dlq_sample": self.dead_letter_queue[:1]
        }

if __name__ == "__main__":
    sample_stream = [
        {"transaction_id": "TXN-90281-ZA", "customer_id": "CUST-104", "amount_zar": 4500.50, "currency": "ZAR", "timestamp": 1719827400},
        {"transaction_id": "TXN-90282-ZA", "customer_id": "CUST-105", "amount_zar": 120.00, "currency": "ZAR", "timestamp": 1719827405},
        # Corrupted / Bad Records
        {"transaction_id": "SHORT", "customer_id": "CUST-106", "amount_zar": -50.0, "currency": "BITCOIN", "timestamp": 1719827410},
        {"customer_id": "CUST-107", "amount_zar": "FOUR_HUNDRED", "currency": "ZAR", "timestamp": 1719827415}
    ]

    pipeline = DataPipelineEngine()
    metrics = pipeline.process_batch(sample_stream)
    print(json.dumps(metrics, indent=2))
