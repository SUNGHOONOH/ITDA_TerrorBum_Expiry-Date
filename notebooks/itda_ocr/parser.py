"""Reviewed post-OCR expiry parser for PaddleOCR fine-tuning experiments.

This module intentionally keeps the public tuple API used by
``expiry_parser_v2.py``.  A parsed value is ``(YYYY, MM, DD)`` and an unknown
component is the literal string ``"NONE"``.  ``final_date`` is the only
serializer needed by the current Colab notebook.

The parser is deterministic. It does not use today's date, invent a month-end
day, or reinterpret ``SELL BY`` as the competition's expiry target.

Project policy: a calendar-valid date-only crop is an expiry candidate even
when it has no nearby ``EXP``/``유통기한`` cue.  A value is rejected only when
its own local context explicitly identifies it as manufacturing, production,
packaging, lot/batch, a duration/example, or ``SELL BY``.  This is essential
because the stage-1 detector intentionally crops just the printed date.
"""

from __future__ import annotations

from collections import Counter
from datetime import date as Date
import re
import unicodedata


PARSER_VERSION = "v4.1.0-realistic-recovery"

# Keep the accepted year window identical to the production parser.  The
# previous 19..39 two-digit window silently rejected 2018 and could then turn
# ``EXP 31/12/18`` into 2031-12-18 through a fallback branch.
YEAR_MIN, YEAR_MAX = 2018, 2040
YY_MIN, YY_MAX = 18, 40
# The competition photographs and the newly collected real set are centred on
# 2026.  Keep this fixed so an identical submission remains reproducible when
# it is rerun in a later calendar year.  It is used only to break ties between
# otherwise calendar-valid interpretations of an unlabelled compact YYMMDD /
# DDMMYY / MMDDYY token; an explicit four-digit year or order cue always wins.
REFERENCE_YEAR = 2026
LATEST_DATE_SCORE_WINDOW = 0.12
MIN_FINAL_REC_CONF = 0.50

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "SEPT": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
_MONTH_NAMES = "|".join(sorted(MONTHS, key=len, reverse=True))

# Order matters only when two role expressions overlap.  SELL BY is kept out
# of POSITIVE on purpose: it describes a retailer action, not the target date.
# English role words may be welded directly to the first date digit by OCR.
# The custom end boundary accepts a following digit/punctuation but still
# rejects longer alphabetic words (for example, SELLBYTE or LOTTERY).
SELL_BY_RE = re.compile(
    r"(?<![A-Z0-9_])SELL[.\s/_-]*BY(?:[.\s/_-]*DATE)?(?=$|[^A-Z_])"
)
KOREAN_EXPIRY_RE = re.compile(
    r"소비\s*기한|유통\s*기한|사용\s*기한|품질\s*유지\s*기한|상미\s*기한|표기일\s*까지|"
    # Public labels also contain the deliberate short forms ``소비`` and
    # ``유통``.  Limit them to a following date/punctuation or end-of-box so
    # ordinary words such as ``소비자`` and ``유통업체`` do not become roles.
    r"(?:소비|유통)(?=\s*(?:[:：./_-]?\s*\d|$))|까지"
)
KOREAN_CONSUMER_EXPIRY_RE = re.compile(r"소비\s*기한")
KOREAN_DISTRIBUTION_EXPIRY_RE = re.compile(r"유통\s*기한")
ENGLISH_EXPIRY_RE = re.compile(
    r"(?<![A-Z0-9_])EXP(?:IRY|IRATION|IRE)?(?:[.\s_-]*DATE)?(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])BSET\s+BEFORE(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])BEST\s*[- ]?(?:BEFORE|BY)(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])BEST\s+IF\s+USED\s+(?:BEFORE|BY)(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])BEFORE(?=\s*[:：./_-]?\s*\d)|"
    r"(?<![A-Z0-9_])USE\s*[- ]?BY(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])BBE(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])BBD(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])BBY(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])B\.?B\.?(?=$|[^A-Z_])"
)
CJK_EXPIRY_RE = re.compile(
    r"有效期限(?:至)?|有效期(?:至)?|有効期限|保质期(?:至)?|保質期(?:至)?|"
    r"到期日|限用日期|截止日期|使用期限|最佳食用日期|"
    r"此\s*日\s*期\s*前\s*最\s*佳|品質保持期限|賞味期限|消費期限"
)
OTHER_EXPIRY_RE = re.compile(
    r"MINDESTENS\s+HALTBAR\s+BIS(?:\s+ENDE)?|"
    r"DA\s+CONSUMARSI\s+PREFERIBILMENTE\s+ENTRO|"
    r"A\s+CONSOMMER\s+DE\s+PREFERENCE\s+AVANT|"
    r"CONSUMIR\s+PREFERENTEMENTE\s+ANTES\s+DEL|"
    r"(?<![A-Z0-9_])(?:T\.?E\.?T\.?T\.?|S\.?K\.?T\.?|S\.?T\.?T\.?|H\.?S\.?D\.?)(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])EXD(?=$|[^A-Z_])|"
    # Public labels contain compact European lot rows such as
    # ``L 095 E 28 10 21``.  Treat the isolated E as an expiry owner only
    # when the following whitespace-separated D M Y row proves DMY by a day
    # greater than 12.  This deliberately leaves ``E 06 04 2026`` unresolved.
    r"(?<![A-Z0-9_])E(?=\s+(?:1[3-9]|2\d|3[01])\s+(?:0?[1-9]|1[0-2])\s+(?:\d{2}|(?:19|20)\d{2})(?:\s|$))|"
    # Imported dot-matrix rows also use a compact P/E pair: P marks the
    # production date and E marks the expiry date.  Accept E only when it is
    # welded to, or immediately followed by, a complete date token so an
    # arbitrary letter inside a batch code cannot become an expiry owner.
    r"(?<![A-Z0-9_])E(?=\s*[:：=]?\s*(?:"
    r"(?:19|20)\d{2}(?:\s*[./_-]\s*|\s+)\d{1,2}(?:\s*[./_-]\s*|\s+)\d{1,2}|"
    r"\d{1,2}(?:\s*[./_-]\s*|\s+)\d{1,2}(?:\s*[./_-]\s*|\s+)(?:19|20)\d{2}|"
    r"\d{6})(?!\d))|"
    r"HẠN\s+SỬ\s+DỤNG|"
    r"(?<![A-Z0-9_])VENCE(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])CAD(?:UCIDAD)?(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])(?:ENDE|FINE|FIN)(?=$|[^A-Z_])|"
    r"(?<![A-Z0-9_])VALID\s+(?:UNTIL|THR(?:OUGH|U))(?=$|[^A-Z_])|"
    r"(?:วัน\s*)?หมด\s*อายุ|(?:วัน|ยา)?\s*สิ้น\s*อายุ|"
    r"ควร\s*บริโภค\s*ก่อน|ใช้\s*ก่อน"
)

# A shelf-life interval is not an absolute endpoint.  Treat the interval as a
# negative owner of any otherwise-unowned date printed immediately after it;
# a nearer explicit EXP/BEST-BEFORE role still owns its own endpoint.
DURATION_RE = re.compile(
    r"(?:保\s*[质質]\s*期(?:\s*限)?|有\s*效\s*期(?:\s*限)?)"
    r"\s*[:：=]?\s*(?:为|為|是)?\s*"
    r"(?:\d{1,3}|[零〇一二两兩三四五六七八九十百]{1,4})\s*"
    r"(?:[个個]\s*)?(?:年|月|日|天)|"
    r"(?:品\s*質\s*保\s*持|賞\s*味)\s*期\s*間\s*"
    r"(?:は\s*)?[:：=]?\s*(?:\d{1,3}|[〇一二三四五六七八九十百]{1,4})\s*"
    r"(?:ヶ|ケ|か|箇|個)?\s*(?:年|月|日)|"
    r"(?<![A-Z0-9_])(?:HSD|HD|HẠN\s+SỬ\s+DỤNG)(?=$|[^A-Z_])"
    r"\s*[:：.=]?\s*(?:L(?:À|A)\s*)?\d{1,3}\s*"
    r"(?:N(?:Ă|A)M|TH(?:Á|A)NG|NG(?:À|A)Y)\b|"
    r"(?<![A-Z0-9_])(?:HSD|HD)(?=$|[^A-Z_])\s*[:：.=]?\s*"
    r"\d{1,3}\s*(?:DAYS?|MONTHS?|YEARS?)\b|"
    r"\b(?:EXPIRY|EXPIRATION|BEST\s+BEFORE|SHELF\s+LIFE)\b"
    r"\s*[:：=]?\s*(?:(?:IS|OF)\s+)?\d{1,3}\s*"
    r"(?:DAYS?|MONTHS?|YEARS?)\b|"
    r"(?:제조일로부터|보관\s*기간|품질\s*유지\s*기간|"
    r"소비\s*기한|유통\s*기한)\s*[:：=]?\s*"
    r"\d{1,3}\s*(?:개\s*)?(?:개월|월|일|년)"
)

# Printed examples describe a format rather than the product's own endpoint.
# Keep the marker bound to a complete expiry-role phrase so ordinary prose
# containing "example", "sample", or the CJK character 例 remains harmless.
_EN_EXPIRY_ROLE_FOR_EXAMPLE = (
    r"(?:EXP(?:IRY|IRATION)?(?:\s*DATE)?|USE\s*(?:BY|BEFORE)|"
    r"BEST\s*(?:BEFORE|BY))"
)
_EN_EXAMPLE_MARKER = r"(?:EXAMPLE|SAMPLE|FORMAT|E\s*\.\s*G\s*\.?)"
_EN_PREFIX_EXAMPLE_MARKER = r"(?:EXAMPLE|SAMPLE|E\s*\.\s*G\s*\.?)"
_CJK_EXPIRY_ROLE_FOR_EXAMPLE = (
    r"(?:保\s*[质質]\s*期(?:\s*限)?|有\s*效\s*期(?:\s*限)?|"
    r"最\s*佳\s*食\s*用\s*日\s*期|賞\s*味\s*期\s*限|"
    r"消\s*費\s*期\s*限|使\s*用\s*期\s*限)"
)
_CJK_EXAMPLE_MARKER = r"(?:示\s*例|[范範]\s*例|例)"
EXPLANATORY_EXAMPLE_RE = re.compile(
    r"(?:소비|유통|사용)\s*기한\s*"
    r"(?:(?:보는|표시|표기)\s*)?예(?:시)?(?![가-힣A-Z])|"
    rf"{_CJK_EXPIRY_ROLE_FOR_EXAMPLE}\s*(?:标\s*示|標\s*示|表\s*示)?\s*"
    rf"{_CJK_EXAMPLE_MARKER}|{_CJK_EXAMPLE_MARKER}[\W_]*{_CJK_EXPIRY_ROLE_FOR_EXAMPLE}|"
    r"(?:賞\s*味|消\s*費|使\s*用|有\s*効|品\s*質\s*保\s*持)"
    r"\s*期\s*限\s*(?:の\s*)?(?:表\s*示\s*)?例|"
    r"(?:V(?:Í|I)\s*D(?:Ụ|U)\s*(?:HSD|HẠN\s+SỬ\s+DỤNG)|"
    r"(?:HSD|HẠN\s+SỬ\s+DỤNG)\s*V(?:Í|I)\s*D(?:Ụ|U))|"
    rf"{_EN_EXPIRY_ROLE_FOR_EXAMPLE}\s*{_EN_EXAMPLE_MARKER}(?:\s*DATE)?|"
    rf"{_EN_PREFIX_EXAMPLE_MARKER}\s*{_EN_EXPIRY_ROLE_FOR_EXAMPLE}"
)

_ASCII_PROCESS_ROLE_TEXT = (
    r"(?<![A-Z0-9_])(?:"
    r"M\s*\.?\s*F\s*\.?\s*[GD](?:[.\s_-]*DATE)?|"
    r"P\s*\.?\s*K\s*\.?\s*D(?:[.\s_-]*DATE)?|"
    r"(?:PRD|PROD)(?:[.\s_-]*DATE)?|"
    r"MANUFACTUR(?:E(?:D)?|ING)(?:\s*(?:DATE|DAY|ON))?|"
    r"DATE\s*OF\s*MANUFACTURE|PRODUCED\s*(?:ON|DATE)|"
    r"PRODUCTION(?:\s*(?:DAY|DATE|DT))?|"
    r"PACK(?:ED(?:\s*(?:DATE|ON))?|ING(?:\s*DATE)?|\s*(?:DATE|ON|DT))|"
    r"PACKAG(?:E|ED|ING)\.?\s*(?:DATE|ON|DT)"
    r")\.?(?=$|[^A-Z_])"
)
_ASCII_AMBIGUOUS_DIRECT_PROCESS_ROLE_TEXT = (
    rf"(?<![A-Z0-9_])(?:PRO|PD|PKG|PACK|MAN|P)\.?(?=[\W_]*(?:\d|(?:{_MONTH_NAMES})(?![A-Z])))"
)
_ASCII_LOT_ROLE_TEXT = (
    r"(?<![A-Z0-9_])(?:"
    r"L\s*\.?\s*O\s*\.?\s*T\s*\.?(?:\s*(?:N\s*\.?\s*O\s*\.?|NUMBER|N[°º]))?|"
    r"BATCH(?:\s*(?:NO\.?|NUMBER|N[°º]))?|"
    r"SERIAL(?:\s*(?:NO\.?|NUMBER|N[°º]))?|S\s*/\s*N"
    r")(?=$|[^A-Z_])"
)
_ASCII_BATCH_SHORTHAND_TEXT = (
    # A real imported row in the collected set prints the expiry first and a
    # batch value as ``B15 07.19.07``.  The leading B owns only this narrow
    # date-shaped code; BEST/BBD words cannot match because of the digit
    # lookahead.
    r"(?<![A-Z0-9_])B(?=\d{1,6}\s+\d{2}[./_-]\d{2}[./_-]\d{2}(?:\D|$))"
)
_KOREAN_PROCESS_ROLE_TEXT = (
    r"(?:제조|생산|포장)(?:\s*(?:연\s*월\s*일|년\s*월\s*일|일자|일|시\s*간))?"
    rf"(?=\s*(?:$|[:：=#/,、;!?._-]*\s*(?:\d|(?:{_MONTH_NAMES})(?![A-Z]))))"
)
_CJK_PROCESS_ROLE_TEXT = (
    r"(?:生\s*[产產産]|(?:制|製)\s*造|包\s*[装裝]|"
    r"灌\s*[装裝]|充\s*填|出\s*[厂廠])"
    r"(?:\s*(?:年\s*月\s*日|日\s*期|日))?"
    rf"(?=\s*(?:$|[:：=#/,、;!?._-]*\s*(?:\d|(?:{_MONTH_NAMES})(?![A-Z]))))|(?:加\s*工|調\s*製)\s*日"
)
_CJK_IDENTIFIER_ROLE_TEXT = (
    r"(?:生\s*[产產]|製\s*造)?\s*批\s*[号號]|"
    r"批\s*次\s*(?:[号號])?|製\s*造\s*(?:番\s*[号號]|ロ\s*ッ\s*ト)|"
    r"ロ\s*ッ\s*ト\s*(?:番\s*[号號]|NO\.?)?"
)
_VI_PROCESS_ROLE_TEXT = (
    r"NG(?:À|A)Y\s+S(?:Ả|A)N\s+XU(?:Ấ|A)T|NG(?:À|A)Y\s+SX|"
    r"NG(?:À|A)Y\s+[ĐD](?:Ó|O)NG\s+G(?:Ó|O)I|"
    r"(?:S(?:Ả|A)N\s+XU(?:Ấ|A)T|[ĐD](?:Ó|O)NG\s+G(?:Ó|O)I)"
    rf"(?=\s*(?:$|[:：=#/,、;!?._-]*\s*(?:\d|(?:{_MONTH_NAMES})(?![A-Z]))))"
)
_VI_IDENTIFIER_ROLE_TEXT = r"S(?:Ố|O)\s*L(?:Ô|O)"
_THAI_PROCESS_ROLE_TEXT = r"(?:วัน(?:ที่)?\s*)?(?:ผลิต|บรรจุ)"
_CJK_PRINTED_PROCESS_INSTRUCTION_TEXT = (
    r"(?:生\s*[产產産]|(?:制|製)\s*造|包\s*[装裝]|"
    r"灌\s*[装裝]|充\s*填|出\s*[厂廠])\s*"
    r"(?:年\s*月\s*日|日\s*期|日)?\s*(?:は\s*)?(?:"
    r"(?:见|見)\s*(?:包\s*[装裝]|外\s*包\s*[装裝]|袋\s*身)"
    r"[^.;。；]{0,12}(?:印\s*[码碼]|打\s*[码碼]|喷\s*[码碼])|"
    r"(?:包\s*[装裝]|容\s*器|枠\s*外|別\s*途)\s*(?:に\s*)?"
    r"(?:記\s*載|印\s*字))"
)
NEGATIVE_RE = re.compile(
    rf"{_KOREAN_PROCESS_ROLE_TEXT}|"
    rf"{_ASCII_PROCESS_ROLE_TEXT}|{_ASCII_AMBIGUOUS_DIRECT_PROCESS_ROLE_TEXT}|"
    rf"{_ASCII_LOT_ROLE_TEXT}|{_ASCII_BATCH_SHORTHAND_TEXT}|"
    rf"{_CJK_PROCESS_ROLE_TEXT}|{_CJK_IDENTIFIER_ROLE_TEXT}|"
    rf"{_VI_PROCESS_ROLE_TEXT}|{_VI_IDENTIFIER_ROLE_TEXT}|{_THAI_PROCESS_ROLE_TEXT}|"
    rf"{_CJK_PRINTED_PROCESS_INSTRUCTION_TEXT}|"
    r"(?<![A-Z0-9_])N\s*\.?\s*S\s*\.?\s*X\.?(?=$|[^A-Z_])|"
    r"품목보고번호"
)
END_RE = re.compile(
    r"(?<![A-Z0-9_])(?:END|ENDE|BBE|FINE|FIN)(?=$|[^A-Z_])|월말|말일|月底"
)
USA_RE = re.compile(
    r"\bUSA?\b|\bU\.S\.A?\.?\b|\bUS\s+FORMAT\b|"
    r"\bMM\s*[/._-]\s*DD\s*[/._-]\s*(?:YY|YYYY)\b|"
    r"\bMMDD(?:YY|YYYY)\b|\bMDY\b"
)
DMY_RE = re.compile(
    r"\bDD\s*[/._-]\s*MM\s*[/._-]\s*(?:YY|YYYY)\b|"
    r"\bDDMM(?:YY|YYYY)\b|\bDMY\b"
)
YMD_RE = re.compile(
    r"\b(?:YY|YYYY)\s*[/._-]\s*MM\s*[/._-]\s*DD\b|"
    r"\b(?:YY|YYYY)MMDD\b|\bYMD\b"
)
MMYY_RE = re.compile(r"\bMM\s*[/._-]\s*(?:YY|YYYY)\b")
YYMM_RE = re.compile(r"\b(?:YY|YYYY)\s*[/._-]\s*MM\b")
TIME_CONTEXT_RE = re.compile(r"\bTIME\b|시간|영업|운영|오픈|마감")

CONFUSION_MAP = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
}

# Colons are deliberately excluded from complete-date separators.  Otherwise
# a valid date followed by a clock (``09.08.2021 08:43``) can be reassembled as
# a different date.  The narrow Korean missing-year rule handles its own colon.
_SEP = r"(?:\s*[./,_\-]{1,3}\s*|\s+)"


def make_date(year: str | int | None, month: str | int, day: str | int):
    """Return ``(YYYY, MM, DD)`` or ``None`` for an invalid calendar date."""
    try:
        month_number, day_number = int(month), int(day)
        if year in (None, "NONE"):
            # Validate month/day without guessing a year.  Leap year 2000 keeps
            # 02-29 available when its year was genuinely unreadable.
            Date(2000, month_number, day_number)
            return ("NONE", f"{month_number:02d}", f"{day_number:02d}")

        year_number = int(year)
        if year_number < 100:
            if not YY_MIN <= year_number <= YY_MAX:
                return None
            year_number += 2000
        if not YEAR_MIN <= year_number <= YEAR_MAX:
            return None
        Date(year_number, month_number, day_number)
        return (str(year_number), f"{month_number:02d}", f"{day_number:02d}")
    except (TypeError, ValueError):
        return None


def make_year_month(year: str | int, month: str | int):
    """Return ``(YYYY, MM, NONE)`` without inventing the month's final day."""
    try:
        year_number, month_number = int(year), int(month)
        if year_number < 100:
            if not YY_MIN <= year_number <= YY_MAX:
                return None
            year_number += 2000
        if not YEAR_MIN <= year_number <= YEAR_MAX or not 1 <= month_number <= 12:
            return None
        return (str(year_number), f"{month_number:02d}", "NONE")
    except (TypeError, ValueError):
        return None


def normalize_ocr_text(text: object) -> str:
    """Normalize Unicode width, Korean units, whitespace and OCR punctuation."""
    normalized = unicodedata.normalize("NFKC", str(text)).upper().replace("|", "/")
    # Date-unit glyphs are separators only when they directly follow a digit.
    # Replacing every 日/月/년 destroys semantic phrases such as 到期日,
    # 生产日期, 製造年月日 and 월말 before role matching.
    normalized = re.sub(r"(?<=\d)\s*년\s*", ".", normalized)
    normalized = re.sub(r"(?<=\d)\s*월\s*", ".", normalized)
    normalized = re.sub(r"(?<=\d)\s*일(?![가-힣])", " ", normalized)
    normalized = re.sub(r"(?<=\d)\s*年\s*", ".", normalized)
    normalized = re.sub(r"(?<=\d)\s*月\s*", ".", normalized)
    normalized = re.sub(r"(?<=\d)\s*日", " ", normalized)
    normalized = re.sub(r"(?<![A-Z])0CT(?![A-Z])", "OCT", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def numeric_ocr_view(text: str) -> str:
    """Correct letter/digit confusions only inside numeric-looking tokens."""
    chars = list(text)
    for index, char in enumerate(chars):
        if char not in CONFUSION_MAP:
            continue
        before = text[index - 1] if index else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        if (before.isdigit() or before in "./,_:-") and (
            after.isdigit() or after in "./,_:-"
        ):
            chars[index] = CONFUSION_MAP[char]
    return "".join(chars)


def _role_matches(text: str):
    patterns = (
        ("sell_by", SELL_BY_RE),
        ("negative", EXPLANATORY_EXAMPLE_RE),
        ("negative", DURATION_RE),
        ("negative", NEGATIVE_RE),
        ("korean", KOREAN_EXPIRY_RE),
        ("cjk", CJK_EXPIRY_RE),
        ("english", ENGLISH_EXPIRY_RE),
        ("other", OTHER_EXPIRY_RE),
    )
    matches = []
    for role, pattern in patterns:
        matches.extend((role, match.span()) for match in pattern.finditer(text))
    return matches


def _has_context_cue(text: str) -> bool:
    return bool(
        _role_matches(text)
        or USA_RE.search(text)
        or DMY_RE.search(text)
        or YMD_RE.search(text)
        or MMYY_RE.search(text)
        or YYMM_RE.search(text)
    )


def _role_for_span(text: str, span, context: str = "") -> str:
    """Return the closest semantic owner of a date span."""
    start, end = span
    if context and EXPLANATORY_EXAMPLE_RE.search(context):
        # A geometry-bound example label owns even an ``EXP:``/``BBE`` prefix
        # OCR emitted inside the sample-value box.
        return "negative"
    matches = [
        (role, role_span)
        for role, role_span in _role_matches(text)
        # A trailing LOT label owns the identifier that follows it, not an
        # already complete date printed to its left (``2022.07.07 I LOT …``).
        # Process-date roles such as MFG remain directional in both positions.
        if not (
            role == "negative"
            and role_span[0] >= end
            and text[role_span[0] : role_span[1]].startswith("LOT")
        )
        # Semantic labels own a value to their right.  The confirmed Korean
        # suffix ``까지`` is the narrow exception (22.01.18 까지).  Letting a
        # later BEST/EXP label reach backwards can steal an earlier example,
        # SELL-BY, manufacture, or lot date on the same OCR line.
        and not (
            role_span[0] >= end
            and not (
                role == "korean"
                and "까지" in text[role_span[0] : role_span[1]]
            )
        )
    ]
    if matches:
        def distance(item):
            _, (role_start, role_end) = item
            if role_end <= start:
                return start - role_end
            if role_start >= end:
                # A suffix such as ``22.01.18 까지`` is a valid owner.
                return role_start - end + 1
            return 0

        role, role_span = min(matches, key=distance)
        if distance((role, role_span)) <= 48:
            return role

    context_matches = _role_matches(context)
    if context_matches:
        # Spatial/sequence context is already selected by proximity, so its
        # final role expression is the most useful one when a label is noisy.
        return max(context_matches, key=lambda item: item[1][1])[0]
    return "unowned"


def _korean_target_kind_for_span(text: str, span, context: str = "") -> str | None:
    """Bind a date to ``소비기한`` or ``유통기한`` for the official priority.

    Role and target kind are intentionally separate: both labels are valid
    expiry owners, but when both are printed the organiser requires the
    consumer-expiry value regardless of chronology.  Prefer the closest
    directional same-line owner, then the already geometry-bound context.
    """

    start, end = span
    patterns = (
        ("consume", KOREAN_CONSUMER_EXPIRY_RE),
        ("distribution", KOREAN_DISTRIBUTION_EXPIRY_RE),
    )
    direct = []
    for kind, pattern in patterns:
        for match in pattern.finditer(text):
            if match.end() <= start:
                direct.append((float(start - match.end()), kind))
    if direct:
        return min(direct, key=lambda item: item[0])[1]

    context_kinds = {
        kind
        for kind, pattern in patterns
        if pattern.search(context) is not None
    }
    return next(iter(context_kinds)) if len(context_kinds) == 1 else None


def _order_policy(text: str, span, role: str, context: str = "") -> str:
    local = f"{context} {text[max(0, span[0] - 48): span[1] + 24]}"
    if USA_RE.search(local):
        return "mdy"
    if YMD_RE.search(local) or role in {"korean", "cjk"}:
        return "ymd"
    if DMY_RE.search(local):
        return "dmy"
    if role in {"english", "other"}:
        return "dmy"
    return "unknown"


def _two_field_policy(text: str, span, role: str, context: str = "") -> str | None:
    local = f"{context} {text[max(0, span[0] - 48): span[1] + 24]}"
    if YYMM_RE.search(local):
        return "yymm"
    if MMYY_RE.search(local):
        return "mmyy"
    if END_RE.search(local) and role in {"korean", "cjk"}:
        return "yymm"
    if END_RE.search(local) and role in {"english", "other"}:
        return "mmyy"
    return None


def _date_only_short_order(token: str, role: str) -> str:
    """Resolve a bare two-digit three-field date without guessing a year.

    Local dot-matrix Korean stamps are predominantly ``YY.MM.DD``. Imported
    date-only crops more often use a slash, space, or hyphen in ``DD/MM/YY``
    order. An explicit role always overrides this typography fallback.
    """
    if role in {"korean", "cjk"}:
        return "ymd"
    if role in {"english", "other"}:
        return "dmy"
    fields = [int(value) for value in re.findall(r"\d{2}", token)]
    if "." in token and len(fields) == 3 and YY_MIN <= fields[0] <= YY_MAX:
        return "ymd"
    if len(fields) == 3 and YY_MIN <= fields[2] <= YY_MAX:
        return "dmy"
    return "ymd" if "." in token else "dmy"


def _three_numeric_values(
    first: str,
    second: str,
    third: str,
    policy: str,
    *,
    locale_fallback: bool = False,
):
    if policy == "ymd":
        primary = make_date(first, second, third)
        if primary is not None or not locale_fallback:
            return [primary]
        # Korean/CJK is a locale prior, not permission to force an impossible
        # calendar date.  Switch only when the remaining calendar-valid order
        # is unique; never choose between two contradictory fallbacks.
        alternatives = (
            make_date(third, second, first),
            make_date(third, first, second),
        )
        unique = []
        for value in alternatives:
            if value is not None and value not in unique:
                unique.append(value)
        return unique if len(unique) == 1 else []
    if policy == "mdy":
        return [make_date(third, first, second)]
    if policy == "dmy":
        primary = make_date(third, second, first)
        # The team convention is English DMY while the middle field can be a
        # month.  If it is >12, the numbers themselves prove MDY instead.
        if primary is None and int(second) > 12 and int(first) <= 12:
            return [make_date(third, first, second)]
        if (
            primary is None
            and YY_MIN <= int(first) <= YY_MAX
            and not YY_MIN <= int(third) <= YY_MAX
        ):
            return [make_date(first, second, third)]
        return [primary]

    # Without a role/order cue, a two-digit value greater than 12 can be either
    # a day or a year.  Enumerate all calendar-valid interpretations and accept
    # only a genuinely unique result.  Returning several equal-score values
    # used to make list order silently choose a locale for date-only crops.
    alternatives = (
        make_date(first, second, third),
        make_date(third, second, first),
        make_date(third, first, second),
    )
    unique = []
    for value in alternatives:
        if value is not None and value not in unique:
            unique.append(value)
    return unique if len(unique) == 1 else []


def _realistic_compact_six_value(first: str, second: str, third: str):
    """Choose the most plausible calendar interpretation of bare six digits.

    A compact token often reaches REC without its printed EXP/locale label.
    Returning NONE for every two-way ambiguity discarded real dates such as
    ``050926`` and ``211125``.  Prefer the valid interpretation whose year is
    closest to the fixed 2026 collection period; for the same year, use the
    non-US DMY convention seen most often in the imported annotations, then
    Korean YMD, and use MDY last.  Explicit role/order cues bypass this helper.
    """
    alternatives = (
        ("dmy", make_date(third, second, first)),
        ("ymd", make_date(first, second, third)),
        ("mdy", make_date(third, first, second)),
    )
    valid = []
    seen = set()
    for priority, (policy, value) in enumerate(alternatives):
        if value is None or value in seen:
            continue
        seen.add(value)
        valid.append((abs(int(value[0]) - REFERENCE_YEAR), priority, policy, value))
    if not valid:
        return None, "unknown"
    _, _, policy, value = min(valid)
    return value, f"{policy}-compact-year-prior"


def _candidate_records(text: object, context: object = ""):
    original_normalized = unicodedata.normalize("NFKC", str(text)).upper()
    context_original = unicodedata.normalize("NFKC", str(context)).upper() if context else ""
    has_cjk_year_month_units = bool(
        re.search(r"\d{2,4}\s*年\s*\d{1,2}\s*月", f"{context_original} {original_normalized}")
    )
    has_korean_year_month_units = bool(
        re.search(r"\d{2,4}\s*년\s*\d{1,2}\s*월", f"{context_original} {original_normalized}")
    )
    normalized = normalize_ocr_text(text)
    context_normalized = normalize_ocr_text(context) if context else ""
    numeric = numeric_ocr_view(normalized)
    records, occupied = [], []

    def overlaps(span):
        return any(max(span[0], start) < min(span[1], end) for start, end in occupied)

    def add(values, span, precision="full", forced_policy=None):
        role = _role_for_span(normalized, span, context_normalized)
        if role == "sell_by":
            return
        # Partial dates are admitted only by the narrow, order-explicit
        # patterns below (four-digit year, printed MM/YY or YY/MM cue,
        # named-month form, or an owned Korean MM.DD stamp).  Do not require
        # an expiry word here: stage-2 date-only crops deliberately omit that
        # context, while the caller is required to pass stage-1 context when
        # process/SELL-BY semantics exist.  Ambiguous bare ``08/21`` still
        # fails closed because ``_two_field_policy`` returns no policy.
        policy = forced_policy or _order_policy(normalized, span, role, context_normalized)
        valid = [tuple(value) for value in values if value is not None]
        for value in valid:
            record = {
                "date": value,
                "span": span,
                "role": role,
                "order_policy": policy,
                "precision": precision,
                "explicit_end": bool(
                    END_RE.search(
                        f"{context_normalized} "
                        f"{normalized[max(0, span[0] - 48): span[1] + 24]}"
                    )
                ),
            }
            if not any(
                existing["date"] == value and existing["span"] == span
                for existing in records
            ):
                records.append(record)
        if valid:
            occupied.append(span)

    def has_strong_expiry_owner(span):
        """Require an explicit expiry owner before repairing a missing year digit."""
        return _role_for_span(normalized, span, context_normalized) in {
            "korean",
            "cjk",
            "english",
            "other",
        }

    # Korean fresh-food labels often print only MM.DD followed immediately by
    # the disposal time.  The hour must never be promoted to a two-digit year:
    # ``03.13.13:47 까지`` means NONE-03-13, not 2013-03-13.  This recovery is
    # owner, and a valid HH:MM suffix. Date-only crops can lack that owner,
    # so unowned stamps are accepted too; explicit process/time owners are not.
    clocked_month_day = re.compile(
        r"(?<![\d./_-])(\d{1,2})[./_-](\d{1,2})[.]?\s*(\d{1,2}):(\d{2})(?!\d)"
    )
    for match in clocked_month_day.finditer(numeric):
        month, day, hour, minute = match.groups()
        role = _role_for_span(normalized, match.span(), context_normalized)
        local = f"{context_normalized} {normalized[max(0, match.start() - 32): match.end() + 20]}"
        valid_clock = 0 <= int(hour) <= 23 and 0 <= int(minute) <= 59
        if not valid_clock:
            continue
        if role in {"negative", "sell_by"} or TIME_CONTEXT_RE.search(local):
            # Reserve the complete MM.DD.HH:MM token.  Without this guard a
            # widened two-digit year pivot can reinterpret the clock hour as
            # a year (02.02.18:17 -> 2018-02-02).  A real expiry date followed
            # by a time has whitespace before HH:MM and does not match here.
            occupied.append(match.span())
            continue
        add(
            [make_date(None, month, day)],
            match.span(),
            precision="month-day",
            forced_policy="md+clock",
        )

    # Named-month forms are order-explicit and therefore locale-independent.
    pattern = rf"(?<![A-Z0-9])(\d{{1,4}}){_SEP}({_MONTH_NAMES}){_SEP}(\d{{1,4}})(?![A-Z0-9])"
    for match in re.finditer(pattern, normalized):
        first, month_name, last = match.groups()
        role = _role_for_span(normalized, match.span(), context_normalized)
        if len(first) == 4:
            values = [make_date(first, MONTHS[month_name], last)]
            named_policy = "ymd-named"
        elif len(last) == 4:
            values = [make_date(last, MONTHS[month_name], first)]
            named_policy = "dmy-named"
        elif len(first) == 2 and len(last) == 2 and role in {"korean", "cjk"}:
            values = [make_date(first, MONTHS[month_name], last)]
            named_policy = "ymd-named-locale"
        elif len(last) == 2:
            values = [make_date(last, MONTHS[month_name], first)]
            named_policy = "dmy-named"
        else:
            values = []
            named_policy = "named-month"
        add(values, match.span(), forced_policy=named_policy)

    pattern = rf"(?<![A-Z0-9])({_MONTH_NAMES}){_SEP}(\d{{1,2}}){_SEP}(\d{{2,4}})(?![A-Z0-9])"
    for match in re.finditer(pattern, normalized):
        month_name, day, year = match.groups()
        add([make_date(year, MONTHS[month_name], day)], match.span(), forced_policy="mdy-named")

    # Imported prints also fuse the two-digit year to a named month while
    # keeping the day separate: 31 JUL21.
    pattern = rf"(?<![A-Z0-9])(\d{{1,2}}){_SEP}({_MONTH_NAMES})(\d{{2,4}})(?![A-Z0-9])"
    for match in re.finditer(pattern, normalized):
        day, month_name, year = match.groups()
        add([make_date(year, MONTHS[month_name], day)], match.span(), forced_policy="dmy-named")

    pattern = rf"(?<![A-Z0-9])(\d{{1,2}})({_MONTH_NAMES})(\d{{2,4}})(?!\d)"
    for match in re.finditer(pattern, normalized):
        day, month_name, year = match.groups()
        add([make_date(year, MONTHS[month_name], day)], match.span(), forced_policy="dmy-named")

    pattern = rf"(?<![A-Z0-9])({_MONTH_NAMES})(\d{{1,2}})((?:19|20)\d{{2}})(?!\d)"
    for match in re.finditer(pattern, normalized):
        month_name, day, year = match.groups()
        add([make_date(year, MONTHS[month_name], day)], match.span(), forced_policy="mdy-named")

    # Month-name/day fused before a slash and full year: DEC01/2021.
    pattern = rf"(?<![A-Z0-9])({_MONTH_NAMES})(\d{{1,2}})[/_-]((?:19|20)\d{{2}})(?!\d)"
    for match in re.finditer(pattern, normalized):
        month_name, day, year = match.groups()
        add([make_date(year, MONTHS[month_name], day)], match.span(), forced_policy="mdy-named")

    # A date followed by a welded OCR letter and a short lot/time suffix.  The
    # required letter prevents this from opening generic product-code matches.
    pattern = rf"(?<!\d)((?:19|20)\d{{2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})[IOQLSB]\d{{1,6}}(?=\D|$)"
    for match in re.finditer(pattern, normalized):
        year, month, day = match.groups()
        add([make_date(year, month, day)], match.span(), forced_policy="ymd")

    # Recover a YYYY-MM-DD stamp when one or two unrelated OCR digits were
    # welded immediately before the year (for example 012026.04./30).
    pattern = rf"(?<!\d)\d{{0,2}}((?:19|20)\d{{2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})(?!\d|\s+\d)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            year, month, day = match.groups()
            add([make_date(year, month, day)], match.span(), forced_policy="ymd")

    # REC sometimes inserts a space inside a two-digit day while retaining a
    # real separator before it (``2022 01 . 2 2``).  Keep this narrow rather
    # than deleting arbitrary digit whitespace, which would merge date fields
    # and lot/time codes.
    pattern = rf"(?<!\d)((?:19|20)\d{{2}}){_SEP}(\d{{1,2}})\s*[./,_-]\s*(\d)\s+(\d)(?!\d)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            year, month, day_tens, day_ones = match.groups()
            add(
                [make_date(year, month, day_tens + day_ones)],
                match.span(),
                forced_policy="ymd-split-day",
            )

    # A single separator may disappear between a one-digit month and a
    # two-digit day (``2026.212`` -> ``2026.2.12``).  This preserves every
    # visible date digit and is deliberately limited to a four-digit year.
    pattern = r"(?<!\d)((?:19|20)\d{2})[./,_-]([1-9])([0-3]\d)(?!\d)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            year, month, day = match.groups()
            add([make_date(year, month, day)], match.span(), forced_policy="ymd-joined-month-day")

    # A Korean expiry stamp can lose the penultimate ``2`` in a 202x year:
    # ``206.2.12까지`` -> ``2026.02.12``.  Recover only this exact 20x shape,
    # only when the token still has a complete valid month/day, and only when
    # an explicit expiry expression owns it.  Bare three-digit years remain
    # rejected instead of being generalized into an unsafe OCR correction.
    pattern = rf"(?<!\d)(20\d){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})(?!\d)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()) and has_strong_expiry_owner(match.span()):
            short_year, month, day = match.groups()
            add(
                [make_date(f"202{short_year[-1]}", month, day)],
                match.span(),
                forced_policy="ymd-missing-2",
            )

    # The same missing-year error can coincide with one missing separator
    # between a one-digit month and two-digit day: ``206.212까지``.
    pattern = r"(?<!\d)(20\d)[./,_-]([1-9])([0-3]\d)(?!\d)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()) and has_strong_expiry_owner(match.span()):
            short_year, month, day = match.groups()
            add(
                [make_date(f"202{short_year[-1]}", month, day)],
                match.span(),
                forced_policy="ymd-missing-2-joined-month-day",
            )

    # Four-digit year first is unambiguous YMD.
    pattern = rf"(?<!\d)((?:19|20)\d{{2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})(?!\d)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            year, month, day = match.groups()
            add([make_date(year, month, day)], match.span(), forced_policy="ymd")

    # Up to three OCR-noise digits may trail an otherwise valid printed day.
    # This recovers examples such as 2026.10.087天 while still validating the
    # first two digits as a calendar day.
    pattern = rf"(?<!\d)((?:19|20)\d{{2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{2}})\d{{1,3}}(?=\D|$)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            year, month, day = match.groups()
            add([make_date(year, month, day)], match.span(), forced_policy="ymd")

    # Four-digit year last: its printed position is stronger evidence than a
    # Korean/CJK YMD locale prior. Explicit order cues and calendar ranges
    # still win. For a date-only crop such as ``03/04/2026``, default to DMY:
    # the external real set uses DMY heavily, while Korean labels normally put
    # the year first. This accepts a visible date instead of returning NONE.
    pattern = rf"(?<!\d)(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}((?:19|20)\d{{2}})(?!\d)"
    for match in re.finditer(pattern, numeric):
        if overlaps(match.span()):
            continue
        first, second, year = match.groups()
        role = _role_for_span(normalized, match.span(), context_normalized)
        local = f"{context_normalized} {normalized[max(0, match.start() - 48): match.end() + 24]}"
        if USA_RE.search(local):
            values = [make_date(year, first, second)]
            selected = "mdy"
        elif DMY_RE.search(local):
            values = [make_date(year, second, first)]
            selected = "dmy"
        elif YMD_RE.search(local):
            # An explicit printed-order declaration contradicts the token.
            values = []
            selected = "ymd-conflict"
        elif int(first) > 12 and int(second) <= 12:
            values = [make_date(year, second, first)]
            selected = "dmy-range"
        elif int(second) > 12 and int(first) <= 12:
            values = [make_date(year, first, second)]
            selected = "mdy-range"
        elif role in {"korean", "cjk", "english", "other"}:
            values = [make_date(year, second, first)]
            selected = "dmy-role"
        else:
            values = [make_date(year, second, first)]
            selected = "dmy-date-only"
        add(values, match.span(), forced_policy=selected)
        if not overlaps(match.span()):
            # Reserve a syntactically complete but calendar-invalid token. A
            # downstream year/month rule must not turn it into a misleading
            # partial value.
            occupied.append(match.span())

    # Three two-digit fields. Explicit USA/role cues win. For date-only crops,
    # typography supplies a documented fallback rather than discarding a
    # visible calendar date: dots -> YY.MM.DD, other separators -> DD/MM/YY.
    pattern = rf"(?<!\d)(\d{{2}}){_SEP}(\d{{2}}){_SEP}(\d{{2}})(?!\d)"
    for match in re.finditer(pattern, numeric):
        if overlaps(match.span()):
            continue
        first, second, third = match.groups()
        # ``12.21 05시`` is a Korean MM.DD stamp followed by a disposal time,
        # not the three-field date 2012-21-05. Leave it for the MD rule.
        if re.match(r"\s*시", normalized[match.end():]):
            continue
        role = _role_for_span(normalized, match.span(), context_normalized)
        policy = _order_policy(normalized, match.span(), role, context_normalized)
        local = f"{context_normalized} {normalized[max(0, match.start() - 48): match.end() + 24]}"
        if policy == "unknown":
            policy = _date_only_short_order(match.group(0), role)
        values = _three_numeric_values(first, second, third, policy)
        if policy == "ymd" and role in {"korean", "cjk"} and not YMD_RE.search(local):
            values = _three_numeric_values(
                first,
                second,
                third,
                policy,
                locale_fallback=True,
            )
        add(
            values,
            match.span(),
            forced_policy=policy,
        )
        if not overlaps(match.span()):
            occupied.append(match.span())

    # Compact DDMMYYYY is a date-only form used in imported product crops.
    # YYYYMMDD was already consumed above, so this only handles year-last
    # values such as 03112021 -> 2021-11-03.
    for match in re.finditer(r"(?<!\d)(\d{2})(\d{2})((?:19|20)\d{2})(?!\d)", numeric):
        if not overlaps(match.span()):
            day, month, year = match.groups()
            add([make_date(year, month, day)], match.span(), forced_policy="dmy-compact")

    # Compact forms are deliberately bounded; long product codes are ignored.
    for match in re.finditer(r"(?<!\d)((?:19|20)\d{6})(?!\d)", numeric):
        if not overlaps(match.span()):
            digits = match.group(1)
            add(
                [make_date(digits[:4], digits[4:6], digits[6:])],
                match.span(),
                forced_policy="ymd",
            )

    for match in re.finditer(r"(?<!\d)(\d{6})(?!\d)", numeric):
        if overlaps(match.span()):
            continue
        digits = match.group(1)
        role = _role_for_span(normalized, match.span(), context_normalized)
        policy = _order_policy(normalized, match.span(), role, context_normalized)
        local = f"{context_normalized} {normalized[max(0, match.start() - 48): match.end() + 24]}"
        locale_fallback = (
            policy == "ymd"
            and role in {"korean", "cjk"}
            and not YMD_RE.search(local)
        )
        if policy == "unknown":
            realistic_value, realistic_policy = _realistic_compact_six_value(
                digits[:2],
                digits[2:4],
                digits[4:],
            )
            values = [realistic_value]
            selected_policy = realistic_policy
        else:
            values = _three_numeric_values(
                digits[:2],
                digits[2:4],
                digits[4:],
                policy,
                locale_fallback=locale_fallback,
            )
            selected_policy = policy
        add(
            values,
            match.span(),
            forced_policy=selected_policy,
        )

    # One separator between a four-digit year and MMDD was lost.
    pattern = rf"(?<!\d)((?:19|20)\d{{2}}){_SEP}(\d{{4}})(?!\d)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            year, month_day = match.groups()
            add(
                [make_date(year, month_day[:2], month_day[2:])],
                match.span(),
                forced_policy="ymd",
            )

    # OCR occasionally joins YYYY and MM before the day separator: 202510.05.
    pattern = r"(?<!\d)((?:19|20)\d{2})(\d{2})[./_-](\d{1,2})(?!\d)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            year, month, day = match.groups()
            add([make_date(year, month, day)], match.span(), forced_policy="ymd")

    # A two-digit YMD stamp may carry one short OCR/lot suffix digit after DD.
    pattern = rf"(?<!\d)(\d{{2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{2}})\d(?=\D|$)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            year, month, day = match.groups()
            add([make_date(year, month, day)], match.span(), forced_policy="ymd")

    # A visible YYYY-MM date is valid even without an expiry cue. The missing
    # day remains NONE; it is never replaced with 28/29/30/31.
    pattern = rf"(?<!\d)((?:19|20)\d{{2}}){_SEP}(\d{{1,2}})[.]?(?![\d./_:\-])"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            year, month = match.groups()
            add([make_year_month(year, month)], match.span(), precision="month", forced_policy="yymm")

    # Explicit Korean/CJK year and month glyphs also disambiguate a one-digit
    # month after a short year (``26년 9월`` / ``26年9月``).  The generic
    # two-short-field rule below intentionally remains fail-closed.
    if has_korean_year_month_units or has_cjk_year_month_units:
        pattern = r"(?<!\d)(\d{2})[.](\d{1,2})[.](?!\d)"
        for match in re.finditer(pattern, numeric):
            if not overlaps(match.span()):
                year, month = match.groups()
                role = _role_for_span(normalized, match.span(), context_normalized)
                if role in {"korean", "cjk", "english", "other"}:
                    add(
                        [make_year_month(year, month)],
                        match.span(),
                        precision="month",
                        forced_policy="yymm",
                    )

    # A visible MM-YYYY date-only stamp is common in the external real crops
    # (for example ``04.2022`` and ``04/2030``).
    pattern = rf"(?<!\d)(\d{{1,2}}){_SEP}((?:19|20)\d{{2}})(?!\d)"
    for match in re.finditer(pattern, numeric):
        if not overlaps(match.span()):
            month, year = match.groups()
            add([make_year_month(year, month)], match.span(), precision="month", forced_policy="mmyy")

    pattern = rf"(?<![A-Z0-9])({_MONTH_NAMES}){_SEP}((?:19|20)\d{{2}})(?!\d)"
    for match in re.finditer(pattern, normalized):
        if not overlaps(match.span()):
            month_name, year = match.groups()
            add(
                [make_year_month(year, MONTHS[month_name])],
                match.span(),
                precision="month",
                forced_policy="named-month",
            )

    # Compact month name plus full year, e.g. ``NOV2021``. Date-only crops do
    # not retain an expiry owner, so the calendar token itself is sufficient.
    pattern = rf"(?<![A-Z0-9])({_MONTH_NAMES})((?:19|20)\d{{2}})(?!\d)"
    for match in re.finditer(pattern, normalized):
        if overlaps(match.span()):
            continue
        month_name, year = match.groups()
        add(
            [make_year_month(year, MONTHS[month_name])],
            match.span(),
            precision="month",
            forced_policy="named-month",
        )

    # Two numeric fields are normally a month/day date in this project. A
    # printed MM-YY/YY-MM/END cue still takes priority and produces a partial
    # month result. With no such cue, choose MD for ``MM-DD`` and DMY only
    # when the first field cannot be a month. Colons stay excluded so clocks
    # such as 14:13 are never promoted to dates.
    pattern = r"(?<![\d./_:\-])(\d{1,2})([./,_-])(\d{1,2})(?![\d./_:\-])"
    for match in re.finditer(pattern, numeric):
        if overlaps(match.span()):
            continue
        first, _, second = match.groups()
        role = _role_for_span(normalized, match.span(), context_normalized)
        preceding = numeric[match.start() - 1] if match.start() else ""
        if role == "unowned" and preceding.isascii() and preceding.isalnum():
            # Do not mine a high-confidence fake date from an OCR/code suffix
            # such as ``F3.1`` or ``U9.20``.  Explicit expiry owners may still
            # be welded directly to a genuine date (for example EXP10.18).
            continue
        policy = _two_field_policy(normalized, match.span(), role, context_normalized)
        if policy is None and role == "cjk" and has_cjk_year_month_units:
            policy = "yymm"
        if policy is None and role == "korean" and has_korean_year_month_units:
            policy = "yymm"
        if policy == "mmyy":
            add([make_year_month(second, first)], match.span(), precision="month", forced_policy=policy)
        elif policy == "yymm":
            add([make_year_month(first, second)], match.span(), precision="month", forced_policy=policy)
        elif 1 <= int(first) <= 12 and 1 <= int(second) <= 31:
            add([make_date(None, first, second)], match.span(), precision="month-day", forced_policy="md-date-only")
        elif 13 <= int(first) <= 31 and 1 <= int(second) <= 12:
            add([make_date(None, second, first)], match.span(), precision="month-day", forced_policy="dmy-date-only")

    return records


def extract_date_candidates(text: object, with_spans: bool = False):
    """Extract calendar-valid tuple candidates from a single OCR string."""
    records = _candidate_records(text)
    if with_spans:
        return [(record["date"], record["span"]) for record in records]
    found = []
    for record in records:
        if record["date"] not in found:
            found.append(record["date"])
    return found


def semantic_score(text: object, span=None, context: object = "") -> float:
    normalized = normalize_ocr_text(text)
    if span is None:
        local = normalized
    else:
        local = normalized[max(0, span[0] - 48): span[1] + 24]
    local = f"{normalize_ocr_text(context) if context else ''} {local}"
    if SELL_BY_RE.search(local):
        return -2.0
    score = 0.0
    if (
        KOREAN_EXPIRY_RE.search(local)
        or CJK_EXPIRY_RE.search(local)
        or ENGLISH_EXPIRY_RE.search(local)
        or OTHER_EXPIRY_RE.search(local)
    ):
        score += 0.32
    if END_RE.search(local):
        score += 0.08
    if NEGATIVE_RE.search(local):
        score -= 0.55
    return score


def _record_semantic_score(record) -> float:
    """Score the already-resolved nearest owner without cross-borrowing roles."""
    role = record.get("role", "unowned")
    if role == "sell_by":
        return -2.0
    if role == "negative":
        return -0.55
    score = 0.32 if role in {"korean", "cjk", "english", "other"} else 0.0
    if record.get("explicit_end"):
        score += 0.08
    return score


def final_date(value) -> str:
    """Serialize the v2-compatible tuple API to the submission-style string."""
    if value is None:
        return "NONE"
    normalized = tuple("NONE" if part is None else str(part) for part in value)
    if normalized == ("NONE", "NONE", "NONE"):
        return "NONE"
    return "-".join(normalized)


def _calendar_order(value):
    """Total ordering that is safe for either None or 'NONE' components."""
    parts = tuple("NONE" if part is None else str(part) for part in value)

    def component(index, missing):
        return missing if parts[index] == "NONE" else int(parts[index])

    return (component(0, -1), component(1, 0), component(2, 0))


def _box_bounds(box):
    if isinstance(box, dict):
        box = box.get("box", box.get("bbox", box.get("points")))
    if box is None:
        return None
    try:
        if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
            x1, y1, x2, y2 = map(float, box)
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        points = [(float(point[0]), float(point[1])) for point in box]
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError, IndexError):
        return None


def _spatial_context(index, texts, boxes):
    target = _box_bounds(boxes[index]) if index < len(boxes) else None
    if target is None:
        return "", None
    tx1, ty1, tx2, ty2 = target
    target_height = max(1.0, ty2 - ty1)
    target_width = max(1.0, tx2 - tx1)
    candidates = []
    for owner_index, owner_text in enumerate(texts):
        if owner_index == index or not _has_context_cue(normalize_ocr_text(owner_text)):
            continue
        owner = _box_bounds(boxes[owner_index]) if owner_index < len(boxes) else None
        if owner is None:
            continue
        ox1, oy1, ox2, oy2 = owner
        owner_height = max(1.0, oy2 - oy1)
        vertical_overlap = max(0.0, min(ty2, oy2) - max(ty1, oy1))
        row_ratio = vertical_overlap / min(target_height, owner_height)
        horizontal_overlap = max(0.0, min(tx2, ox2) - max(tx1, ox1))
        column_ratio = horizontal_overlap / min(target_width, max(1.0, ox2 - ox1))

        if row_ratio >= 0.30 and ox2 <= tx2:
            gap = max(0.0, tx1 - ox2)
            if gap <= 8.0 * max(target_height, owner_height):
                candidates.append((gap / max(target_height, owner_height), owner_index, owner_text))
        elif column_ratio >= 0.25 and oy2 <= ty1:
            gap = ty1 - oy2
            # Example labels should bind only their immediately following
            # sample row.  Their semantics must not leak two rows down into a
            # real BEST/EXP stamp.
            max_vertical_gap = (
                1.5
                if EXPLANATORY_EXAMPLE_RE.search(normalize_ocr_text(owner_text))
                else 3.0
            )
            if gap <= max_vertical_gap * max(target_height, owner_height):
                candidates.append((1.0 + gap / max(target_height, owner_height), owner_index, owner_text))
    if not candidates:
        return "", None
    _, owner_index, owner_text = min(candidates, key=lambda item: item[0])
    return str(owner_text), owner_index


def _recognition_fields(recognition):
    if isinstance(recognition, dict):
        return (
            str(recognition.get("text", recognition.get("transcription", ""))),
            float(recognition.get("rec_conf", recognition.get("score", 1.0))),
        )
    return str(recognition), 1.0


def scored_candidates(recognitions, rois=None, variant_names=None, boxes=None):
    """Build selector-ready candidates; ``boxes`` is an optional v3 addition.

    Existing v2 calls using only recognitions/rois/variant_names remain valid.
    A box may be ``[x1, y1, x2, y2]``, a four-point polygon, or a dictionary
    containing ``box``, ``bbox`` or ``points``.
    """
    recognitions = list(recognitions)
    rois = list(rois) if rois is not None else [{} for _ in recognitions]
    if len(rois) < len(recognitions):
        rois.extend({} for _ in range(len(recognitions) - len(rois)))
    variant_names = list(variant_names) if variant_names is not None else ["raw"] * len(recognitions)
    if len(variant_names) < len(recognitions):
        variant_names.extend("raw" for _ in range(len(recognitions) - len(variant_names)))

    texts = [_recognition_fields(recognition)[0] for recognition in recognitions]
    if boxes is None:
        boxes = [
            roi.get("box", roi.get("bbox", roi.get("points"))) if isinstance(roi, dict) else None
            for roi in rois
        ]
    else:
        boxes = list(boxes)
        if len(boxes) < len(recognitions):
            boxes.extend(None for _ in range(len(recognitions) - len(boxes)))
    geometry_available = any(_box_bounds(box) is not None for box in boxes)

    candidates = []
    for index, recognition in enumerate(recognitions):
        text, rec_conf = _recognition_fields(recognition)
        roi = rois[index]
        variant = variant_names[index]
        det_conf = float(roi.get("det_conf", roi.get("score", 1.0))) if isinstance(roi, dict) else 1.0
        roi_index = int(roi.get("roi_index", index)) if isinstance(roi, dict) else index

        context, owner_index = _spatial_context(index, texts, boxes)
        if (
            not context
            and not geometry_available
            and not _role_matches(normalize_ocr_text(text))
        ):
            # PaddleOCR commonly emits a role line immediately before its date
            # line.  This conservative adjacency fallback keeps text-only
            # notebook callers useful without requiring geometry.
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(texts) and _has_context_cue(normalize_ocr_text(texts[neighbor])):
                    context, owner_index = texts[neighbor], neighbor
                    break

        records = _candidate_records(text, context=context)
        # The same physical ROI is commonly recognized through raw/Otsu/color
        # variants.  Date occurrence order is stable enough to identify the
        # same printed slot without putting OCR text (which varies) in the
        # source identity.  This prevents a lower-confidence later year from
        # receiving a second-source vote merely because its transcription is
        # different, while preserving two distinct dates printed in one ROI.
        ordered_spans = sorted({record["span"] for record in records})
        span_slots = {span: slot for slot, span in enumerate(ordered_spans)}
        for parse_rank, record in enumerate(records):
            parsed, span = record["date"], record["span"]
            score = rec_conf + 0.10 * det_conf + _record_semantic_score(record)
            score += 0.09 if record["precision"] == "full" else 0.0
            score += 0.03 if record["order_policy"] != "unknown" else 0.0
            score -= 0.01 * parse_rank
            candidates.append(
                {
                    "date": parsed,
                    "score": float(score),
                    "text": text,
                    "context": context,
                    "context_owner_index": owner_index,
                    "rec_conf": rec_conf,
                    "det_conf": det_conf,
                    "roi_index": roi_index,
                    "variant": variant,
                    "span": span,
                    "role": record["role"],
                    "target_kind": _korean_target_kind_for_span(
                        normalize_ocr_text(text),
                        span,
                        normalize_ocr_text(context),
                    ),
                    "order_policy": record["order_policy"],
                    "precision": record["precision"],
                    "explicit_end": record["explicit_end"],
                    "box": boxes[index] if index < len(boxes) else None,
                    "source_key": (roi_index, span_slots[span]),
                }
            )

    # A confirmed OCR split writes ``소비기한 2027.07.`` and ``20까지``
    # as two boxes.  Join only this exact incomplete-left / suffix-owned-day
    # shape.  With geometry, the boxes must share a nearby row; without it,
    # only immediate list neighbours are eligible.  Arbitrary OCR strings are
    # never concatenated.
    left_fragment = re.compile(
        r"(?<!\d)((?:19|20)\d{2})\s*[./_-]\s*(\d{1,2})\s*[./_-]\s*$"
    )
    right_fragment = re.compile(r"^\s*(\d{1,2})\s*까지\s*$")

    def fragment_boxes_are_near(left_box, right_box):
        left = _box_bounds(left_box)
        right = _box_bounds(right_box)
        if left is None or right is None:
            return False
        lx1, ly1, lx2, ly2 = left
        rx1, ry1, rx2, ry2 = right
        left_height = max(1.0, ly2 - ly1)
        right_height = max(1.0, ry2 - ry1)
        left_width = max(1.0, lx2 - lx1)
        right_width = max(1.0, rx2 - rx1)
        vertical_overlap = max(0.0, min(ly2, ry2) - max(ly1, ry1))
        row_ratio = vertical_overlap / min(left_height, right_height)
        horizontal_gap = max(0.0, rx1 - lx2)
        same_row = (
            row_ratio >= 0.30
            and rx1 >= lx1
            and horizontal_gap <= 3.0 * max(left_height, right_height)
        )
        horizontal_overlap = max(0.0, min(lx2, rx2) - max(lx1, rx1))
        column_ratio = horizontal_overlap / min(left_width, right_width)
        vertical_gap = max(0.0, ry1 - ly2)
        next_row = (
            ry1 >= ly1
            and column_ratio >= 0.30
            and vertical_gap <= 1.5 * max(left_height, right_height)
        )
        return same_row or next_row

    for index in range(len(texts) - 1):
        left_text = normalize_ocr_text(texts[index])
        right_text = normalize_ocr_text(texts[index + 1])
        left_match = left_fragment.search(left_text)
        right_match = right_fragment.fullmatch(right_text)
        if left_match is None or right_match is None:
            continue
        if geometry_available and not fragment_boxes_are_near(
            boxes[index], boxes[index + 1]
        ):
            continue

        expected = make_date(
            left_match.group(1),
            left_match.group(2),
            right_match.group(1),
        )
        if expected is None:
            continue
        combined = f"{texts[index]} {texts[index + 1]}"
        record = next(
            (
                item
                for item in _candidate_records(combined)
                if item["date"] == expected and item["precision"] == "full"
            ),
            None,
        )
        if record is None:
            continue

        _, left_rec_conf = _recognition_fields(recognitions[index])
        _, right_rec_conf = _recognition_fields(recognitions[index + 1])
        left_roi, right_roi = rois[index], rois[index + 1]
        left_det_conf = (
            float(left_roi.get("det_conf", left_roi.get("score", 1.0)))
            if isinstance(left_roi, dict)
            else 1.0
        )
        right_det_conf = (
            float(right_roi.get("det_conf", right_roi.get("score", 1.0)))
            if isinstance(right_roi, dict)
            else 1.0
        )
        left_roi_index = (
            int(left_roi.get("roi_index", index))
            if isinstance(left_roi, dict)
            else index
        )
        right_roi_index = (
            int(right_roi.get("roi_index", index + 1))
            if isinstance(right_roi, dict)
            else index + 1
        )
        score = min(left_rec_conf, right_rec_conf)
        score += 0.10 * min(left_det_conf, right_det_conf)
        score += _record_semantic_score(record) + 0.09
        if record["order_policy"] != "unknown":
            score += 0.03
        candidates.append(
            {
                "date": expected,
                "score": float(score),
                "text": combined,
                "context": "",
                "context_owner_index": None,
                "rec_conf": min(left_rec_conf, right_rec_conf),
                "det_conf": min(left_det_conf, right_det_conf),
                "roi_index": left_roi_index,
                "variant": "fragment-join",
                "span": record["span"],
                "role": record["role"],
                "target_kind": _korean_target_kind_for_span(
                    normalize_ocr_text(combined),
                    record["span"],
                ),
                "order_policy": record["order_policy"],
                "precision": "full",
                "explicit_end": record["explicit_end"],
                "box": [boxes[index], boxes[index + 1]],
                "source_key": (
                    "fragment",
                    left_roi_index,
                    right_roi_index,
                ),
            }
        )
    return candidates


def choose_candidate(candidates):
    """Choose one v2-style candidate dictionary, or return ``None``."""
    if not candidates:
        return None

    usable = [candidate for candidate in candidates if candidate.get("role") != "sell_by"]
    if not usable:
        return None
    # A manufacture/lot-only image is not a valid expiry answer.
    if all(candidate.get("role") == "negative" for candidate in usable):
        return None

    # Official semantic priority: if both labels are recognized, select among
    # consumer-expiry candidates and never let the generic later-date rule
    # replace them with a distribution-expiry date.
    if any(
        candidate.get("target_kind") == "consume"
        and candidate.get("role") == "korean"
        for candidate in usable
    ):
        usable = [
            candidate
            for candidate in usable
            if not (
                candidate.get("target_kind") == "distribution"
                and candidate.get("role") == "korean"
            )
        ]

    base_scored = []
    for candidate in usable:
        value = tuple(candidate["date"])
        score = float(candidate.get("score", 0.0))
        if all(part not in (None, "NONE") for part in value):
            score += 0.04
        base_scored.append((candidate, score))

    # Resolve order alternatives within one recognized token before voting or
    # selecting the chronologically latest date.
    source_winners = {}
    for candidate, score in base_scored:
        key = candidate.get(
            "source_key",
            (candidate.get("roi_index", 0), candidate.get("text", ""), candidate.get("span")),
        )
        incumbent = source_winners.get(key)
        if incumbent is None or score > incumbent[1]:
            source_winners[key] = (candidate, score)

    # Consensus is counted only after preprocessing alternatives from the same
    # physical source have been collapsed.  Raw/Otsu/color views of one box are
    # correlated readings, not independent votes.
    votes = Counter(
        tuple(candidate["date"])
        for candidate, _ in source_winners.values()
    )
    ranked = [
        (
            candidate,
            score + 0.04 * min(2, votes[tuple(candidate["date"])] - 1),
        )
        for candidate, score in source_winners.values()
    ]
    top_candidate, top_score = max(ranked, key=lambda item: item[1])

    def accept(candidate):
        """Apply the final OCR-confidence abstention gate when supplied."""
        return (
            candidate
            if float(candidate.get("rec_conf", 1.0)) >= MIN_FINAL_REC_CONF
            else None
        )

    # Only apply the "later of two dates" rule between similarly reliable,
    # complete, expiry-owned values.  Partial dates and MFG/LOT values cannot
    # enter this calendar comparison, so NONE never reaches int().
    competitive = [
        (candidate, score)
        for candidate, score in ranked
        if candidate.get("role", "unowned") in {"korean", "cjk", "english", "other", "unowned"}
        and all(part not in (None, "NONE") for part in candidate["date"])
        and score >= top_score - LATEST_DATE_SCORE_WINDOW
    ]
    distinct_dates = {tuple(candidate["date"]) for candidate, _ in competitive}
    distinct_sources = {candidate.get("source_key") for candidate, _ in competitive}
    if len(distinct_dates) >= 2 and len(distinct_sources) >= 2:
        latest = max(_calendar_order(candidate["date"]) for candidate, _ in competitive)
        latest_candidates = [
            item for item in competitive if _calendar_order(item[0]["date"]) == latest
        ]
        return accept(max(latest_candidates, key=lambda item: item[1])[0])

    # Apply the same later-date rule to partial dates only when every semantic
    # and precision signal agrees.  In particular, compare NONE-MM-DD only
    # with NONE-MM-DD (or YYYY-MM-NONE only with the same shape), never against
    # a complete date or a different expiry role.  This handles adjacent
    # date-only stamps such as 10.13 and 10.14 without inventing a year.
    top_value = tuple(top_candidate["date"])
    top_missing = tuple(part in (None, "NONE") for part in top_value)
    if any(top_missing) and not all(top_missing):
        top_tier = (
            top_candidate.get("role", "unowned"),
            top_candidate.get("target_kind"),
            top_candidate.get("precision"),
            top_missing,
        )
        partial_competitive = [
            (candidate, score)
            for candidate, score in ranked
            if candidate.get("role", "unowned")
            in {"korean", "cjk", "english", "other", "unowned"}
            and (
                candidate.get("role", "unowned"),
                candidate.get("target_kind"),
                candidate.get("precision"),
                tuple(part in (None, "NONE") for part in candidate["date"]),
            )
            == top_tier
            and score >= top_score - LATEST_DATE_SCORE_WINDOW
        ]
        partial_dates = {
            tuple(candidate["date"]) for candidate, _ in partial_competitive
        }
        partial_sources = {
            candidate.get("source_key") for candidate, _ in partial_competitive
        }
        if len(partial_dates) >= 2 and len(partial_sources) >= 2:
            latest = max(
                _calendar_order(candidate["date"])
                for candidate, _ in partial_competitive
            )
            latest_candidates = [
                item
                for item in partial_competitive
                if _calendar_order(item[0]["date"]) == latest
            ]
            return accept(max(latest_candidates, key=lambda item: item[1])[0])

    return accept(top_candidate)


def parse_expiry_candidates(texts, rec_confs=None, det_confs=None, boxes=None):
    """Return scored candidates; the first three parameters match v2."""
    texts = list(texts)
    rec_confs = list(rec_confs) if rec_confs is not None else [1.0] * len(texts)
    det_confs = list(det_confs) if det_confs is not None else [1.0] * len(texts)
    boxes = list(boxes) if boxes is not None else None
    recognitions = [
        {
            "text": text,
            "rec_conf": rec_confs[index] if index < len(rec_confs) else 1.0,
        }
        for index, text in enumerate(texts)
    ]
    rois = [
        {
            "roi_index": index,
            "det_conf": det_confs[index] if index < len(det_confs) else 1.0,
            "box": boxes[index] if boxes is not None and index < len(boxes) else None,
        }
        for index in range(len(texts))
    ]
    return scored_candidates(recognitions, rois, boxes=boxes)


def choose_expiry_date(texts, rec_confs=None, det_confs=None, boxes=None):
    """Return the selected candidate dictionary, or ``None``."""
    return choose_candidate(parse_expiry_candidates(texts, rec_confs, det_confs, boxes))


PARSER_ACCEPT = {
    "2027.06.26": "2027-06-26",
    "EXP: APR-28-2023": "2023-04-28",
    "USE BY 15FEB22B": "2022-02-15",
    "BEST BY 02OCT2022": "2022-10-02",
    "2027년 04월 12일까지": "2027-04-12",
    "EXP 27/03/21": "2021-03-27",
    "USA EXP 01/09/27": "2027-01-09",
    "BEST BEFORE END: 08 2021": "2021-08-NONE",
    "MINDESTENS HALTBAR BIS ENDE: 12.20": "2020-12-NONE",
    # Verified against local date-only crops.
    "10.18 11:24": "NONE-10-18",
    "10.15까지 13:56": "NONE-10-15",
    "10.22까지": "NONE-10-22",
    "04.2022": "2022-04-NONE",
    "04/2030": "2030-04-NONE",
    "03/04/2026": "2026-04-03",
    "NOV2021": "2021-11-NONE",
    "26.03.31": "2026-03-31",
    "30/07/26": "2026-07-30",
    "03112021": "2021-11-03",
    "2026.212까지": "2026-02-12",
    "206.212까지": "2026-02-12",
    "205.12.04까지": "2025-12-04",
    "(DDMMYY) 050926": "2026-09-05",
    # Bare compact dates use the fixed 2026 collection-period prior instead
    # of being discarded merely because two calendar orders are possible.
    "050926": "2026-09-05",
    "211125 7788": "2025-11-21",
    "261231": "2026-12-31",
    "301223": "2023-12-30",
    "121623": "2023-12-16",
    # Full dates must survive adjacent time, line, quantity and lot numbers.
    "2027.04.12 까지 2W 19:35": "2027-04-12",
    "BEST BY FEB/06/21 07:45 FL6A": "2021-02-06",
    "2022.07.07 I LOT 01895001": "2022-07-07",
    "소비기한 2026.09.19 까지 내용량 300g 355kcal": "2026-09-19",
    "071125 H1 / NO4 27 A": "2025-11-07",
    # A compact imported production/expiry pair must select E, not P.
    "BNE 1348 P09.10.2025 E03.04.2028": "2028-04-03",
    "HSD / EXP: 070527 B15 07.19.07": "2027-05-07",
}
PARSER_REJECT = (
    "SELL BY 12/31/26",
    "20130628332176",
    "8801055721955",
    "14:13",
    "348 KCAL",
    "제조시간 10.18",
    "제조일 2026.06.10",
    "LOT 10.18",
    "05:44 F3.1",
    "08.U9.20 A",
    "205.12.04",
    "P09.10.2025",
    "B15 07.19.07",
)


def run_self_tests():
    for text, expected in PARSER_ACCEPT.items():
        winner = choose_expiry_date([text])
        actual = final_date(winner["date"]) if winner else "NONE"
        assert actual == expected, (text, expected, actual)
    for text in PARSER_REJECT:
        winner = choose_expiry_date([text])
        actual = final_date(winner["date"]) if winner else "NONE"
        assert actual == "NONE", (text, actual)
    winner = choose_expiry_date(
        ["10.13", "10.14 09:45 P"],
        rec_confs=[0.9983, 0.9537],
    )
    actual = final_date(winner["date"]) if winner else "NONE"
    assert actual == "NONE-10-14", ("partial-latest", actual)
    winner = choose_expiry_date(["B 20 2 020"], rec_confs=[0.3198])
    assert winner is None, ("low-confidence-abstention", winner)
    winner = choose_expiry_date(
        ["BNE 1348 P09.10.2025", "E03.04.2028"],
    )
    actual = final_date(winner["date"]) if winner else "NONE"
    assert actual == "2028-04-03", ("split-production-expiry", actual)
    return f"{PARSER_VERSION}: accept {len(PARSER_ACCEPT)}, reject {len(PARSER_REJECT)} passed"


if __name__ == "__main__":
    print(run_self_tests())
