"""
Compliance Filter for Safe Scout.

Provides:
- Runtime content compliance checking
- Paywall and login wall detection
- Malicious content detection
- Copyright and license compliance
- Content quality filtering
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Overall compliance status."""
    COMPLIANT = "compliant"
    BLOCKED = "blocked"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"


class ComplianceViolation(Enum):
    """Types of compliance violations."""
    NONE = "none"
    PAYWALL_DETECTED = "paywall_detected"
    LOGIN_REQUIRED = "login_required"
    CAPTCHA_REQUIRED = "captcha_required"
    AGE_GATE = "age_gate"
    COOKIE_WALL = "cookie_wall"
    GEOGRAPHIC_BLOCK = "geographic_block"
    RATE_LIMITED = "rate_limited"
    ACCESS_DENIED = "access_denied"
    COPYRIGHT_RESTRICTED = "copyright_restricted"
    MALICIOUS_CONTENT = "malicious_content"
    LOW_QUALITY = "low_quality"
    ADULT_CONTENT = "adult_content"
    SCRAPING_BLOCKED = "scraping_blocked"


@dataclass
class ComplianceResult:
    """Result of compliance check."""
    url: str
    status: ComplianceStatus
    violations: List[ComplianceViolation] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    confidence: float = 1.0
    content_quality_score: float = 1.0
    checked_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_compliant(self) -> bool:
        return self.status == ComplianceStatus.COMPLIANT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "status": self.status.value,
            "violations": [v.value for v in self.violations],
            "messages": self.messages,
            "confidence": self.confidence,
            "content_quality_score": self.content_quality_score,
            "checked_at": self.checked_at.isoformat(),
            "metadata": self.metadata,
        }


# Paywall detection patterns
PAYWALL_PATTERNS = [
    # Common paywall text
    r"subscribe\s+(?:now\s+)?to\s+(?:continue\s+)?read",
    r"sign\s+up\s+to\s+read",
    r"create\s+(?:a\s+)?free\s+account\s+to\s+continue",
    r"this\s+(?:article|content)\s+is\s+for\s+(?:premium\s+)?(?:subscribers|members)\s+only",
    r"unlock\s+this\s+(?:article|content)",
    r"(?:premium|subscriber)\s+(?:only|exclusive)\s+content",
    r"free\s+trial\s+to\s+continue\s+reading",
    r"article\s+limit\s+reached",
    r"you(?:'ve|\s+have)\s+reached\s+your\s+(?:free\s+)?(?:article|viewing)\s+limit",
    r"register\s+to\s+read\s+more",
    r"already\s+a\s+subscriber\?\s+(?:sign\s+)?in",

    # Payment prompts
    r"(?:start\s+)?your\s+(?:free\s+)?(?:trial|subscription)",
    r"(?:only\s+)?\$\d+(?:\.\d+)?(?:\s*\/\s*(?:mo|month|year))",
    r"(?:get\s+)?unlimited\s+access\s+for",
    r"become\s+a\s+(?:premium\s+)?member",
]

LOGIN_PATTERNS = [
    r"(?:please\s+)?(?:sign|log)\s*in\s+to\s+(?:continue|view|access)",
    r"you\s+must\s+be\s+logged\s+in",
    r"login\s+required",
    r"authentication\s+required",
    r"sign\s+in\s+with\s+(?:google|github|facebook|twitter)",
    r"create\s+an\s+account\s+to\s+continue",
    r"session\s+(?:has\s+)?expired",
]

CAPTCHA_PATTERNS = [
    r"verify\s+you(?:'re|\s+are)\s+(?:a\s+)?human",
    r"complete\s+(?:the\s+)?captcha",
    r"recaptcha",
    r"hcaptcha",
    r"prove\s+you(?:'re|\s+are)\s+not\s+a\s+(?:robot|bot)",
    r"security\s+check",
    r"challenge\s+required",
]

RATE_LIMIT_PATTERNS = [
    r"too\s+many\s+requests",
    r"rate\s+limit(?:ed)?",
    r"slow\s+down",
    r"try\s+again\s+(?:in\s+)?\d+\s+(?:seconds?|minutes?)",
    r"request\s+limit\s+exceeded",
    r"temporarily\s+blocked",
]

ACCESS_DENIED_PATTERNS = [
    r"403\s+forbidden",
    r"access\s+denied",
    r"you\s+don(?:'t|\s+not)\s+have\s+permission",
    r"unauthorized\s+access",
    r"blocked\s+by\s+(?:cloudflare|firewall)",
    r"ip\s+(?:address\s+)?(?:has\s+been\s+)?blocked",
]

SCRAPING_BLOCKED_PATTERNS = [
    r"automated\s+(?:access|requests?)\s+(?:detected|blocked)",
    r"bot\s+(?:detected|blocked)",
    r"scraping\s+(?:is\s+)?(?:not\s+)?allowed",
    r"please\s+use\s+(?:our\s+)?api",
    r"disable\s+(?:your\s+)?(?:ad\s*)?blocker",
]

COPYRIGHT_PATTERNS = [
    r"all\s+rights\s+reserved",
    r"(?:no\s+)?reproduction\s+(?:is\s+)?permitted",
    r"(?:do\s+not|cannot)\s+(?:copy|reproduce|distribute)",
    r"protected\s+by\s+copyright",
    r"proprietary\s+(?:and\s+)?confidential",
]

ADULT_CONTENT_PATTERNS = [
    r"(?:18\+|21\+)\s+only",
    r"adult\s+content",
    r"(?:explicit|mature)\s+(?:material|content)",
    r"must\s+be\s+(?:18|21)\s+(?:years?\s+old|or\s+older)",
    r"age\s+verification\s+required",
]


@dataclass
class ComplianceFilterConfig:
    """Configuration for compliance filter."""
    # Detection sensitivity
    paywall_confidence_threshold: float = 0.7
    login_confidence_threshold: float = 0.7

    # Content quality
    min_content_length: int = 200  # Minimum chars for valid content
    max_boilerplate_ratio: float = 0.7  # Max ratio of boilerplate to content

    # Strictness
    block_on_warning: bool = False
    require_explicit_copyright_license: bool = False

    # Custom patterns
    additional_block_patterns: List[str] = field(default_factory=list)
    additional_allow_patterns: List[str] = field(default_factory=list)


class ComplianceFilter:
    """
    Filters content for compliance with legal and ethical guidelines.

    Detects:
    - Paywalls and subscription walls
    - Login requirements
    - CAPTCHA/anti-bot measures
    - Rate limiting responses
    - Copyright restrictions
    - Low-quality/spam content
    """

    def __init__(
        self,
        config: Optional[ComplianceFilterConfig] = None,
    ):
        self.config = config or ComplianceFilterConfig()

        # Compile patterns
        self._paywall_patterns = [
            re.compile(p, re.IGNORECASE) for p in PAYWALL_PATTERNS
        ]
        self._login_patterns = [
            re.compile(p, re.IGNORECASE) for p in LOGIN_PATTERNS
        ]
        self._captcha_patterns = [
            re.compile(p, re.IGNORECASE) for p in CAPTCHA_PATTERNS
        ]
        self._rate_limit_patterns = [
            re.compile(p, re.IGNORECASE) for p in RATE_LIMIT_PATTERNS
        ]
        self._access_denied_patterns = [
            re.compile(p, re.IGNORECASE) for p in ACCESS_DENIED_PATTERNS
        ]
        self._scraping_blocked_patterns = [
            re.compile(p, re.IGNORECASE) for p in SCRAPING_BLOCKED_PATTERNS
        ]
        self._copyright_patterns = [
            re.compile(p, re.IGNORECASE) for p in COPYRIGHT_PATTERNS
        ]
        self._adult_patterns = [
            re.compile(p, re.IGNORECASE) for p in ADULT_CONTENT_PATTERNS
        ]

        # Custom patterns
        self._custom_block_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.config.additional_block_patterns
        ]
        self._custom_allow_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.config.additional_allow_patterns
        ]

    async def check_compliance(
        self,
        url: str,
        html_content: str,
        text_content: str,
        http_status: int,
        response_headers: Dict[str, str],
    ) -> ComplianceResult:
        """
        Check content for compliance violations.

        Args:
            url: The URL being checked
            html_content: Raw HTML content
            text_content: Extracted text content
            http_status: HTTP response status code
            response_headers: HTTP response headers

        Returns:
            ComplianceResult with violation details
        """
        violations: List[ComplianceViolation] = []
        messages: List[str] = []
        metadata: Dict[str, Any] = {}

        # Check HTTP status first
        status_violation = self._check_http_status(http_status)
        if status_violation:
            violations.append(status_violation[0])
            messages.append(status_violation[1])

        # Check for paywall
        paywall_result = self._detect_paywall(html_content, text_content)
        if paywall_result[0]:
            violations.append(ComplianceViolation.PAYWALL_DETECTED)
            messages.append(f"Paywall detected (confidence: {paywall_result[1]:.2f})")
            metadata["paywall_confidence"] = paywall_result[1]
            metadata["paywall_indicators"] = paywall_result[2]

        # Check for login requirement
        login_result = self._detect_login_wall(html_content, text_content)
        if login_result[0]:
            violations.append(ComplianceViolation.LOGIN_REQUIRED)
            messages.append(f"Login required (confidence: {login_result[1]:.2f})")
            metadata["login_confidence"] = login_result[1]

        # Check for CAPTCHA
        if self._detect_captcha(html_content, text_content):
            violations.append(ComplianceViolation.CAPTCHA_REQUIRED)
            messages.append("CAPTCHA or anti-bot challenge detected")

        # Check for rate limiting
        if self._detect_rate_limit(html_content, text_content, response_headers):
            violations.append(ComplianceViolation.RATE_LIMITED)
            messages.append("Rate limiting or throttling detected")

        # Check for scraping block
        if self._detect_scraping_block(html_content, text_content):
            violations.append(ComplianceViolation.SCRAPING_BLOCKED)
            messages.append("Anti-scraping measures detected")

        # Check for access denial
        if self._detect_access_denied(html_content, text_content):
            violations.append(ComplianceViolation.ACCESS_DENIED)
            messages.append("Access denied by server")

        # Check for adult content
        if self._detect_adult_content(html_content, text_content):
            violations.append(ComplianceViolation.ADULT_CONTENT)
            messages.append("Adult content gate detected")

        # Check content quality
        quality_score = self._assess_content_quality(text_content)
        metadata["content_quality_score"] = quality_score

        if quality_score < 0.3:
            violations.append(ComplianceViolation.LOW_QUALITY)
            messages.append(f"Low content quality (score: {quality_score:.2f})")

        # Check custom block patterns
        for pattern in self._custom_block_patterns:
            if pattern.search(html_content) or pattern.search(text_content):
                violations.append(ComplianceViolation.MALICIOUS_CONTENT)
                messages.append(f"Matched custom block pattern: {pattern.pattern}")

        # Check custom allow patterns (can override blocks)
        has_allow_override = False
        for pattern in self._custom_allow_patterns:
            if pattern.search(html_content) or pattern.search(text_content):
                has_allow_override = True
                messages.append(f"Matched allow override pattern: {pattern.pattern}")
                break

        # Determine overall status
        if has_allow_override and not any(
            v in [ComplianceViolation.MALICIOUS_CONTENT, ComplianceViolation.ADULT_CONTENT]
            for v in violations
        ):
            status = ComplianceStatus.COMPLIANT
            violations = []
        elif any(
            v in [
                ComplianceViolation.PAYWALL_DETECTED,
                ComplianceViolation.LOGIN_REQUIRED,
                ComplianceViolation.CAPTCHA_REQUIRED,
                ComplianceViolation.ACCESS_DENIED,
                ComplianceViolation.MALICIOUS_CONTENT,
                ComplianceViolation.ADULT_CONTENT,
            ]
            for v in violations
        ):
            status = ComplianceStatus.BLOCKED
        elif violations:
            status = (
                ComplianceStatus.BLOCKED
                if self.config.block_on_warning
                else ComplianceStatus.WARNING
            )
        else:
            status = ComplianceStatus.COMPLIANT

        return ComplianceResult(
            url=url,
            status=status,
            violations=violations,
            messages=messages,
            confidence=1.0 - (len(violations) * 0.1),
            content_quality_score=quality_score,
            metadata=metadata,
        )

    def _check_http_status(
        self,
        status_code: int,
    ) -> Optional[Tuple[ComplianceViolation, str]]:
        """Check HTTP status code for issues."""
        if status_code == 401:
            return (
                ComplianceViolation.LOGIN_REQUIRED,
                "HTTP 401 Unauthorized"
            )
        elif status_code == 403:
            return (
                ComplianceViolation.ACCESS_DENIED,
                "HTTP 403 Forbidden"
            )
        elif status_code == 429:
            return (
                ComplianceViolation.RATE_LIMITED,
                "HTTP 429 Too Many Requests"
            )
        elif status_code == 451:
            return (
                ComplianceViolation.GEOGRAPHIC_BLOCK,
                "HTTP 451 Unavailable For Legal Reasons"
            )
        elif status_code >= 400:
            return (
                ComplianceViolation.ACCESS_DENIED,
                f"HTTP {status_code} Error"
            )

        return None

    def _detect_paywall(
        self,
        html: str,
        text: str,
    ) -> Tuple[bool, float, List[str]]:
        """Detect paywall presence."""
        combined = f"{html}\n{text}".lower()
        indicators: List[str] = []

        for pattern in self._paywall_patterns:
            matches = pattern.findall(combined)
            if matches:
                indicators.extend(matches[:2])  # Max 2 per pattern

        if not indicators:
            return (False, 0.0, [])

        # Calculate confidence based on number of indicators
        confidence = min(1.0, len(indicators) * 0.3)

        return (
            confidence >= self.config.paywall_confidence_threshold,
            confidence,
            indicators[:5],  # Return top 5 indicators
        )

    def _detect_login_wall(
        self,
        html: str,
        text: str,
    ) -> Tuple[bool, float]:
        """Detect login requirement."""
        combined = f"{html}\n{text}".lower()
        match_count = 0

        for pattern in self._login_patterns:
            if pattern.search(combined):
                match_count += 1

        if match_count == 0:
            return (False, 0.0)

        confidence = min(1.0, match_count * 0.35)

        return (
            confidence >= self.config.login_confidence_threshold,
            confidence,
        )

    def _detect_captcha(self, html: str, text: str) -> bool:
        """Detect CAPTCHA presence."""
        combined = f"{html}\n{text}".lower()

        for pattern in self._captcha_patterns:
            if pattern.search(combined):
                return True

        # Also check for common CAPTCHA element IDs/classes
        captcha_elements = [
            "g-recaptcha",
            "h-captcha",
            "cf-turnstile",
            "captcha-container",
            "captcha_container",
        ]

        for elem in captcha_elements:
            if elem in html.lower():
                return True

        return False

    def _detect_rate_limit(
        self,
        html: str,
        text: str,
        headers: Dict[str, str],
    ) -> bool:
        """Detect rate limiting."""
        # Check headers
        rate_limit_headers = [
            "x-ratelimit-remaining",
            "x-rate-limit-remaining",
            "retry-after",
        ]

        for header in rate_limit_headers:
            if header.lower() in {k.lower() for k in headers.keys()}:
                remaining = headers.get(header, "")
                if remaining and remaining.isdigit() and int(remaining) == 0:
                    return True

        # Check content
        combined = f"{html}\n{text}".lower()

        for pattern in self._rate_limit_patterns:
            if pattern.search(combined):
                return True

        return False

    def _detect_scraping_block(self, html: str, text: str) -> bool:
        """Detect anti-scraping measures."""
        combined = f"{html}\n{text}".lower()

        for pattern in self._scraping_blocked_patterns:
            if pattern.search(combined):
                return True

        return False

    def _detect_access_denied(self, html: str, text: str) -> bool:
        """Detect access denial."""
        combined = f"{html}\n{text}".lower()

        for pattern in self._access_denied_patterns:
            if pattern.search(combined):
                return True

        return False

    def _detect_adult_content(self, html: str, text: str) -> bool:
        """Detect adult content gates."""
        combined = f"{html}\n{text}".lower()

        for pattern in self._adult_patterns:
            if pattern.search(combined):
                return True

        return False

    def _assess_content_quality(self, text: str) -> float:
        """Assess content quality (0-1 score)."""
        if not text:
            return 0.0

        # Length check
        if len(text) < self.config.min_content_length:
            return 0.1

        # Check for actual sentences
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [
            s.strip() for s in sentences
            if len(s.strip()) > 20  # Min sentence length
        ]

        if len(valid_sentences) < 2:
            return 0.2

        # Check for code/technical content (positive signal)
        code_patterns = [
            r'```',
            r'def\s+\w+\s*\(',
            r'function\s+\w+\s*\(',
            r'class\s+\w+',
            r'import\s+\w+',
            r'const\s+\w+\s*=',
        ]

        has_code = any(
            re.search(p, text) for p in code_patterns
        )

        # Check for boilerplate ratio
        boilerplate_patterns = [
            r'cookie\s+(?:policy|consent)',
            r'privacy\s+policy',
            r'terms\s+(?:of\s+)?(?:service|use)',
            r'newsletter',
            r'subscribe\s+to\s+our',
            r'follow\s+us\s+on',
            r'share\s+(?:this\s+)?(?:on|via)',
            r'(?:read|view)\s+more\s+articles',
        ]

        boilerplate_count = sum(
            1 for p in boilerplate_patterns
            if re.search(p, text, re.IGNORECASE)
        )

        # Calculate score
        score = 0.5

        # Add for content length (up to +0.2)
        length_bonus = min(0.2, len(text) / 10000)
        score += length_bonus

        # Add for sentence count (up to +0.15)
        sentence_bonus = min(0.15, len(valid_sentences) / 50)
        score += sentence_bonus

        # Add for code content (+0.15)
        if has_code:
            score += 0.15

        # Subtract for boilerplate (up to -0.3)
        boilerplate_penalty = min(0.3, boilerplate_count * 0.05)
        score -= boilerplate_penalty

        return max(0.0, min(1.0, score))

    async def check_batch(
        self,
        items: List[Dict[str, Any]],
        max_concurrent: int = 5,
    ) -> List[ComplianceResult]:
        """Check compliance for multiple items."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def check_with_limit(item: Dict[str, Any]) -> ComplianceResult:
            async with semaphore:
                return await self.check_compliance(
                    url=item["url"],
                    html_content=item.get("html", ""),
                    text_content=item.get("text", ""),
                    http_status=item.get("status", 200),
                    response_headers=item.get("headers", {}),
                )

        results = await asyncio.gather(
            *[check_with_limit(item) for item in items]
        )

        return list(results)


# Convenience exports
__all__ = [
    "ComplianceFilter",
    "ComplianceFilterConfig",
    "ComplianceResult",
    "ComplianceStatus",
    "ComplianceViolation",
]
