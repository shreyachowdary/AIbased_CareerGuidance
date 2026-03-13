"""
Password validation for registration.
"""

import re
from typing import List, Tuple


def validate_password(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password strength.
    Returns (is_valid, list of error messages).
    """
    errors = []
    if len(password) < 8:
        errors.append("At least 8 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("At least one uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("At least one lowercase letter")
    if not re.search(r"\d", password):
        errors.append("At least one number")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("At least one special character (!@#$%^&*)")
    return len(errors) == 0, errors
