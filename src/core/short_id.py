"""Short ID generation utilities for LeanVibe Agent Hive 2.0."""

import hashlib
import string
from typing import Optional


class ShortIDGenerator:
    """Generate short, human-friendly IDs for tasks and agents."""

    # Character sets for different types of IDs
    ALPHANUMERIC = string.ascii_lowercase + string.digits
    AGENT_CHARS = string.ascii_lowercase  # agents use letters only
    TASK_CHARS = string.digits  # tasks use numbers only

    @staticmethod
    def generate_agent_short_id(name: str, uuid_str: str) -> str:
        """Generate a short ID for an agent based on name and UUID.

        Format: 3-letter abbreviation from name + 2 chars from UUID hash
        Example: "meta" -> "met" + "7a" -> "met7a"
        """
        # Get 3-letter abbreviation from name
        name_clean = "".join(c for c in name.lower() if c.isalpha())
        if len(name_clean) >= 3:
            name_part = name_clean[:3]
        else:
            name_part = name_clean.ljust(3, "x")

        # Get 2 chars from UUID hash
        uuid_hash = hashlib.sha256(uuid_str.encode()).hexdigest()
        uuid_part = "".join(c for c in uuid_hash if c in ShortIDGenerator.ALPHANUMERIC)[
            :2
        ]

        return f"{name_part}{uuid_part}"

    @staticmethod
    def generate_task_short_id(title: str, uuid_str: str) -> str:
        """Generate a short ID for a task based on title and UUID.

        Format: 4-digit number from title hash + UUID hash
        Example: "Fix bug" -> "3847"
        """
        # Combine title and UUID for uniqueness
        combined = f"{title}{uuid_str}"
        hash_obj = hashlib.sha256(combined.encode())
        hash_hex = hash_obj.hexdigest()

        # Extract digits and create 4-digit ID
        digits = "".join(c for c in hash_hex if c.isdigit())
        if len(digits) >= 4:
            return digits[:4]

        # Fallback: pad with hash chars converted to numbers
        result = digits
        for char in hash_hex:
            if len(result) >= 4:
                break
            if char in string.ascii_lowercase:
                # Convert a-f to 0-5
                if char in "abcdef":
                    result += str(ord(char) - ord("a"))

        return result[:4].ljust(4, "0")

    @staticmethod
    def generate_session_short_id(name: str, uuid_str: str) -> str:
        """Generate a short ID for a session.

        Format: 2-letter abbreviation + 3 chars from hash
        Example: "dev-session" -> "ds" + "4k2" -> "ds4k2"
        """
        # Get 2-letter abbreviation
        words = name.lower().replace("-", " ").replace("_", " ").split()
        if len(words) >= 2:
            name_part = words[0][0] + words[1][0]
        elif len(words) == 1 and len(words[0]) >= 2:
            name_part = words[0][:2]
        else:
            name_part = "ss"  # default for session

        # Get 3 chars from UUID hash
        uuid_hash = hashlib.sha256(uuid_str.encode()).hexdigest()
        hash_part = "".join(c for c in uuid_hash if c in ShortIDGenerator.ALPHANUMERIC)[
            :3
        ]

        return f"{name_part}{hash_part}"

    @staticmethod
    def is_valid_short_id(short_id: str, id_type: str) -> bool:
        """Validate a short ID format."""
        if not short_id:
            return False

        if id_type == "agent":
            return (
                len(short_id) == 5 and short_id[:3].isalpha() and short_id[3:].isalnum()
            )
        elif id_type == "task":
            return len(short_id) == 4 and short_id.isdigit()
        elif id_type == "session":
            return (
                len(short_id) == 5 and short_id[:2].isalpha() and short_id[2:].isalnum()
            )

        return False

    @staticmethod
    def search_short_id_candidates(partial_id: str, id_type: str) -> list[str]:
        """Generate possible completions for partial short IDs."""
        candidates = []

        if id_type == "agent" and len(partial_id) <= 5:
            # For agents, try common prefixes
            common_prefixes = ["met", "arc", "dev", "qat", "sys"]
            for prefix in common_prefixes:
                if prefix.startswith(partial_id):
                    candidates.append(prefix)

        return candidates


def generate_short_id(name: str, uuid_str: str, id_type: str) -> str:
    """Generate a short ID based on type."""
    generator = ShortIDGenerator()

    if id_type == "agent":
        return generator.generate_agent_short_id(name, uuid_str)
    elif id_type == "task":
        return generator.generate_task_short_id(name, uuid_str)
    elif id_type == "session":
        return generator.generate_session_short_id(name, uuid_str)
    else:
        raise ValueError(f"Unknown ID type: {id_type}")


def resolve_short_id(short_id: str, id_type: str) -> Optional[str]:
    """Resolve a short ID to a full UUID (requires database lookup)."""
    # This will be implemented in the database layer
    # Returns None if not found, UUID string if found
    pass
