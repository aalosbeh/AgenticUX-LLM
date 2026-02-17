"""
Privacy Manager for Agentic UX System
Privacy-preserving data handling and user consent management.
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """Categories of personal data"""
    BEHAVIORAL = "behavioral"  # clicks, navigation
    BIOMETRIC = "biometric"  # heart rate, pupil dilation
    LOCATION = "location"
    DEVICE = "device"  # browser, OS
    PERSONAL = "personal"  # name, email
    PERFORMANCE = "performance"  # task metrics


class ConsentLevel(Enum):
    """User consent levels"""
    NONE = "none"
    BASIC = "basic"  # Only essential behavioral data
    ANALYTICS = "analytics"  # Behavioral + aggregate analytics
    FULL = "full"  # All data including biometric
    EXPLICIT_DENIAL = "explicit_denial"


@dataclass
class ConsentRecord:
    """User consent record"""
    user_id: str
    consent_level: ConsentLevel
    timestamp: str
    expires_at: str
    purposes: List[str]  # personalization, research, analytics
    data_categories: List[str]
    ip_address_hashed: str
    user_agent_hashed: str


@dataclass
class RetentionPolicy:
    """Data retention policy"""
    data_category: DataCategory
    retention_days: int
    anonymize_after_days: Optional[int] = None
    delete_on_user_request: bool = True


class PrivacyManager:
    """
    Manages privacy compliance and user consent.
    Implements data minimization, anonymization, and retention policies.
    """

    def __init__(self):
        self.user_consents: Dict[str, ConsentRecord] = {}
        self.consent_history: Dict[str, List[ConsentRecord]] = {}
        self.retention_policies = self._initialize_policies()
        self.data_minimization_rules = self._initialize_minimization_rules()
        self.sensitive_patterns = self._initialize_sensitive_patterns()
        self.user_deletion_queue: Set[str] = set()

    def _initialize_policies(self) -> Dict[DataCategory, RetentionPolicy]:
        """Initialize default retention policies"""
        return {
            DataCategory.BEHAVIORAL: RetentionPolicy(
                data_category=DataCategory.BEHAVIORAL,
                retention_days=90,
                anonymize_after_days=30
            ),
            DataCategory.BIOMETRIC: RetentionPolicy(
                data_category=DataCategory.BIOMETRIC,
                retention_days=7,
                anonymize_after_days=1
            ),
            DataCategory.PERSONAL: RetentionPolicy(
                data_category=DataCategory.PERSONAL,
                retention_days=365,
                anonymize_after_days=None
            ),
            DataCategory.DEVICE: RetentionPolicy(
                data_category=DataCategory.DEVICE,
                retention_days=90,
                anonymize_after_days=30
            ),
            DataCategory.PERFORMANCE: RetentionPolicy(
                data_category=DataCategory.PERFORMANCE,
                retention_days=180,
                anonymize_after_days=90
            ),
        }

    def _initialize_minimization_rules(self) -> Dict[DataCategory, List[str]]:
        """Initialize data minimization rules - what data to collect"""
        return {
            DataCategory.BEHAVIORAL: [
                "click_count",
                "time_spent",
                "task_completion",
                "navigation_patterns"
            ],
            DataCategory.BIOMETRIC: [
                "heart_rate_aggregate",  # Only aggregates, not raw
                "pupil_dilation_trends"  # Only trends, not continuous
            ],
            DataCategory.DEVICE: [
                "browser_type",
                "os_version",
                "viewport_size"
            ],
            DataCategory.PERSONAL: [
                "user_id_hashed",
                "age_range",  # Not exact age
                "general_location"  # Not precise
            ],
        }

    def _initialize_sensitive_patterns(self) -> Dict[str, str]:
        """Initialize patterns for detecting sensitive data"""
        return {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "address": r"\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)",
        }

    def request_user_consent(
        self,
        user_id: str,
        purposes: List[str],
        ip_address: str,
        user_agent: str
    ) -> Dict[str, Any]:
        """
        Request user consent for data processing.
        Returns consent request with options.
        """
        return {
            "request_id": f"consent_{user_id}_{int(datetime.utcnow().timestamp())}",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "purposes": purposes,
            "consent_options": [
                {
                    "level": ConsentLevel.BASIC.value,
                    "description": "Essential features only",
                    "data_categories": [DataCategory.BEHAVIORAL.value]
                },
                {
                    "level": ConsentLevel.ANALYTICS.value,
                    "description": "Enable analytics and improvements",
                    "data_categories": [
                        DataCategory.BEHAVIORAL.value,
                        DataCategory.DEVICE.value,
                        DataCategory.PERFORMANCE.value
                    ]
                },
                {
                    "level": ConsentLevel.FULL.value,
                    "description": "All features including personalization",
                    "data_categories": [c.value for c in DataCategory]
                }
            ],
            "duration_days": 365
        }

    def record_consent(
        self,
        user_id: str,
        consent_level: ConsentLevel,
        purposes: List[str],
        ip_address: str,
        user_agent: str,
        duration_days: int = 365
    ) -> bool:
        """Record user consent"""
        ip_hash = self._hash_string(ip_address)
        ua_hash = self._hash_string(user_agent)

        record = ConsentRecord(
            user_id=user_id,
            consent_level=consent_level,
            timestamp=datetime.utcnow().isoformat(),
            expires_at=(datetime.utcnow() + timedelta(days=duration_days)).isoformat(),
            purposes=purposes,
            data_categories=[c.value for c in DataCategory] if consent_level == ConsentLevel.FULL else
                           [DataCategory.BEHAVIORAL.value] if consent_level == ConsentLevel.BASIC else
                           [DataCategory.BEHAVIORAL.value, DataCategory.DEVICE.value],
            ip_address_hashed=ip_hash,
            user_agent_hashed=ua_hash
        )

        self.user_consents[user_id] = record

        if user_id not in self.consent_history:
            self.consent_history[user_id] = []
        self.consent_history[user_id].append(record)

        logger.info(f"Recorded consent for {user_id}: {consent_level.value}")
        return True

    def check_consent(self, user_id: str, data_category: DataCategory) -> bool:
        """Check if user has consented to collect data category"""
        if user_id not in self.user_consents:
            return False

        record = self.user_consents[user_id]

        # Check if consent expired
        if datetime.fromisoformat(record.expires_at) < datetime.utcnow():
            return False

        if record.consent_level == ConsentLevel.NONE or record.consent_level == ConsentLevel.EXPLICIT_DENIAL:
            return False

        return data_category.value in record.data_categories

    def sanitize_data(
        self,
        data: Dict[str, Any],
        user_id: str,
        data_category: DataCategory
    ) -> Dict[str, Any]:
        """
        Remove or anonymize sensitive data based on user consent
        and data minimization rules.
        """
        if not self.check_consent(user_id, data_category):
            return {}

        sanitized = {}
        allowed_fields = self.data_minimization_rules.get(data_category.value, [])

        for field, value in data.items():
            # Only include allowed fields
            if field not in allowed_fields:
                continue

            # Check for sensitive patterns
            if isinstance(value, str):
                if self._contains_sensitive_data(value):
                    value = self._anonymize_value(value)
                elif len(value) > 100:
                    # Truncate long strings
                    value = value[:100]

            sanitized[field] = value

        return sanitized

    def _contains_sensitive_data(self, value: str) -> bool:
        """Check if value contains sensitive data"""
        for pattern in self.sensitive_patterns.values():
            if re.search(pattern, value):
                return True
        return False

    def _anonymize_value(self, value: str) -> str:
        """Anonymize a sensitive value"""
        # Hash sensitive value
        hashed = hashlib.sha256(value.encode()).hexdigest()[:16]
        return f"[ANONYMIZED_{hashed}]"

    def anonymize_user_data(self, user_id: str, data_age_days: int = 30) -> int:
        """
        Anonymize user data older than threshold.
        Returns count of anonymized records.
        """
        # In production, this would iterate through database
        # For now, return mock count
        logger.info(f"Anonymizing data for {user_id} older than {data_age_days} days")
        return 0

    def request_data_deletion(self, user_id: str) -> Dict[str, Any]:
        """
        Process user's right to be forgotten.
        Initiates data deletion process.
        """
        self.user_deletion_queue.add(user_id)

        logger.info(f"Processing deletion request for {user_id}")

        return {
            "status": "deletion_requested",
            "user_id": user_id,
            "request_date": datetime.utcnow().isoformat(),
            "estimated_completion": (datetime.utcnow() + timedelta(days=30)).isoformat(),
            "message": "Your data deletion request has been submitted. We will complete this within 30 days."
        }

    def execute_data_deletion(self, user_id: str) -> bool:
        """
        Execute data deletion for user.
        Removes all personal data in compliance with regulations.
        """
        if user_id not in self.user_deletion_queue:
            return False

        # Delete from various stores
        if user_id in self.user_consents:
            del self.user_consents[user_id]
        if user_id in self.consent_history:
            del self.consent_history[user_id]

        self.user_deletion_queue.remove(user_id)

        logger.info(f"Deleted all data for {user_id}")
        return True

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Export all user data in machine-readable format.
        Implements data portability right.
        """
        export = {
            "user_id": user_id,
            "export_date": datetime.utcnow().isoformat(),
            "consent_records": []
        }

        if user_id in self.consent_history:
            export["consent_records"] = [
                asdict(record) for record in self.consent_history[user_id]
            ]

        logger.info(f"Exported data for {user_id}")
        return export

    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        return {
            "report_date": datetime.utcnow().isoformat(),
            "users_with_consent": len(self.user_consents),
            "consent_distribution": self._get_consent_distribution(),
            "deletion_queue_size": len(self.user_deletion_queue),
            "retention_policies": {
                k.value: {
                    "retention_days": v.retention_days,
                    "anonymize_after_days": v.anonymize_after_days
                }
                for k, v in self.retention_policies.items()
            }
        }

    def _get_consent_distribution(self) -> Dict[str, int]:
        """Get distribution of consent levels"""
        distribution = {level.value: 0 for level in ConsentLevel}

        for record in self.user_consents.values():
            distribution[record.consent_level.value] += 1

        return distribution

    def _hash_string(self, value: str) -> str:
        """Hash a string for privacy"""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get privacy manager statistics"""
        return {
            "users_consented": len(self.user_consents),
            "consent_history_size": sum(len(h) for h in self.consent_history.values()),
            "pending_deletions": len(self.user_deletion_queue),
            "retention_policies_count": len(self.retention_policies)
        }


if __name__ == "__main__":
    manager = PrivacyManager()

    # Request and record consent
    user_id = "user_001"

    # Request consent
    request = manager.request_user_consent(
        user_id=user_id,
        purposes=["personalization", "analytics"],
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0..."
    )

    print("Consent request:")
    print(f"  Options: {len(request['consent_options'])}")

    # Record consent
    manager.record_consent(
        user_id=user_id,
        consent_level=ConsentLevel.ANALYTICS,
        purposes=["personalization", "analytics"],
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0..."
    )

    # Check consent
    can_collect_behavioral = manager.check_consent(user_id, DataCategory.BEHAVIORAL)
    can_collect_biometric = manager.check_consent(user_id, DataCategory.BIOMETRIC)

    print(f"\nConsent check:")
    print(f"  Behavioral data: {can_collect_behavioral}")
    print(f"  Biometric data: {can_collect_biometric}")

    # Sanitize data
    raw_data = {
        "click_count": 42,
        "user_email": "user@example.com",
        "task_time": 120
    }

    sanitized = manager.sanitize_data(raw_data, user_id, DataCategory.BEHAVIORAL)
    print(f"\nSanitized data: {sanitized}")

    # Export user data
    export = manager.export_user_data(user_id)
    print(f"\nExported {len(export['consent_records'])} consent records")

    # Privacy report
    report = manager.generate_privacy_report()
    print(f"\nPrivacy report:")
    print(f"  Users with consent: {report['users_with_consent']}")
    print(f"  Consent distribution: {report['consent_distribution']}")
