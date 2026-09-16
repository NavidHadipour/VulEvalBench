"""The MITRE CWE hierarchy, used to decide when two CWE ids refer to the same
weakness at different levels of specificity.

WHY THIS EXISTS
    The benchmark's catalogue names sixteen CWE identifiers. Detectors do not
    all label findings at the same granularity: Bandit reports broad classes,
    while Semgrep's rules frequently report a more specific child. Scoring by
    exact string match therefore penalises a tool for being *more precise*.

    A concrete case from a real run: Semgrep reported CWE-276 "Incorrect
    Default Permissions" on a sample whose intended category was CWE-732
    "Incorrect Permission Assignment for Critical Resource". MITRE lists
    CWE-276 as `ChildOf` CWE-732 -- the tool had found exactly the right
    weakness -- yet exact matching discarded the finding.

    Exact matching is therefore not a neutral policy. It systematically favours
    tools that label coarsely. This module supplies the alternative.

DESCENDANTS ONLY, NEVER ANCESTORS
    A finding counts if its CWE is a catalogue entry *or a descendant of one*.
    Ancestors deliberately do not count. Accepting them would mean a tool
    answering CWE-20 "Improper Input Validation" -- a catch-all parent of a
    large part of the tree -- was credited for every injection flaw in the
    benchmark. Descendants narrow the claim; ancestors broaden it, and only
    narrowing preserves the meaning of the category.

PROVENANCE
    The relationships are imported wholesale from MITRE's published catalogue,
    not hand-selected. That distinction matters: this rule was added after
    noticing it disadvantaged one tool, so the mapping must be decided by an
    external authority rather than by anyone watching the scores. Re-import at
    any time with:

        python src/cwe_taxonomy.py

    The result is cached in data/cwe_hierarchy.json with the catalogue version
    recorded, so a run remains reproducible without network access.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path

from common import read_json, write_json

CWE_XML_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
DATA_DIR = Path(__file__).resolve().parent / "data"
HIERARCHY_FILE = DATA_DIR / "cwe_hierarchy.json"


def _strip_namespace(tag: str) -> str:
    """ElementTree keeps namespaces as `{uri}tag`; only the local name matters."""
    return tag.rsplit("}", 1)[-1]


def build_hierarchy(zip_path: Path) -> dict:
    """Parse MITRE's catalogue archive into a parent -> children mapping.

    Args:
        zip_path: the downloaded `cwec_latest.xml.zip`.

    Returns:
        A dict with the catalogue `version` and a `children` map keyed by
        parent CWE id, each value a list of direct child ids.
    """
    archive = zipfile.ZipFile(zip_path)
    # The archive holds one file, named for the version it contains.
    xml_name = next(n for n in archive.namelist() if n.endswith(".xml"))
    root = ElementTree.parse(io.BytesIO(archive.read(xml_name))).getroot()

    version = root.attrib.get("Version", "unknown")
    children: dict[str, list[str]] = {}

    for element in root.iter():
        if _strip_namespace(element.tag) != "Weakness":
            continue
        weakness_id = element.attrib.get("ID")
        if not weakness_id:
            continue

        for relation in element.iter():
            if _strip_namespace(relation.tag) != "Related_Weakness":
                continue
            # Only ChildOf edges define the specificity hierarchy. CanPrecede,
            # PeerOf and similar natures describe other kinds of association
            # and must not be treated as "the same weakness, more precisely".
            if relation.attrib.get("Nature") != "ChildOf":
                continue
            parent = relation.attrib.get("CWE_ID")
            if parent:
                children.setdefault(parent, []).append(weakness_id)

    # De-duplicate: a weakness declares the same parent once per CWE view.
    children = {p: sorted(set(c)) for p, c in children.items()}
    return {"version": version, "source": CWE_XML_URL, "children": children}


def load_hierarchy() -> dict:
    """Load the cached hierarchy, or an empty one if it has not been built.

    Returns an empty mapping rather than raising, so a checkout without the
    cached file still runs -- it simply falls back to exact CWE matching.
    """
    if not HIERARCHY_FILE.exists():
        return {"version": "unavailable", "children": {}}
    return read_json(HIERARCHY_FILE)


_NUMBER_RE = re.compile(r"(\d+)")


def _number(cwe: str) -> str | None:
    """Extract the numeric part of a `CWE-89`-style identifier."""
    match = _NUMBER_RE.search(cwe or "")
    return match.group(1) if match else None


def descendants_of(cwe_ids: list[str], hierarchy: dict | None = None) -> set[str]:
    """Every CWE at or below any of `cwe_ids`, as `CWE-nnn` strings.

    Walks the ChildOf tree transitively, so a grandchild counts as well as a
    direct child. The `seen` set makes the walk safe against the cycles that
    MITRE's multi-view relationships can introduce.
    """
    hierarchy = hierarchy or load_hierarchy()
    children = hierarchy.get("children", {})

    seen: set[str] = set()
    stack = [n for n in (_number(c) for c in cwe_ids) if n]

    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children.get(current, []))

    return {f"CWE-{n}" for n in seen}


def main() -> None:
    """Download MITRE's catalogue and rebuild the cached hierarchy."""
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "cwec_latest.xml.zip"

    if not zip_path.exists():
        print(f"Downloading {CWE_XML_URL}")
        urllib.request.urlretrieve(CWE_XML_URL, zip_path)

    hierarchy = build_hierarchy(zip_path)
    write_json(HIERARCHY_FILE, hierarchy)

    print(f"CWE catalogue version : {hierarchy['version']}")
    print(f"Parents with children : {len(hierarchy['children'])}")
    print(f"Written to            : {HIERARCHY_FILE}")


if __name__ == "__main__":
    main()
