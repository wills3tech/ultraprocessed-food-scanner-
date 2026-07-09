"""
Ultra-Processed Food (UPF) Scanner
Flask backend for Vercel serverless deployment.

Responsibilities:
1. Serve the single-page frontend.
2. Look up barcodes against the Open Food Facts API and normalize the response.
3. Classify NOVA group (1-4) either from Open Food Facts data directly, or via
   a heuristic ingredient-text classifier used as a fallback for the OCR/AI flow.
"""

import re
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

import requests

OFF_API_BASE = "https://world.openfoodfacts.org/api/v2/product"
OFF_TIMEOUT_SECONDS = 8

# ---------------------------------------------------------------------------
# NOVA classification metadata
# ---------------------------------------------------------------------------

NOVA_METADATA = {
    1: {
        "label": "Unprocessed / Minimally Processed",
        "description": (
            "Natural foods altered only by processes such as removal of "
            "inedible parts, drying, crushing, or pasteurization. No added "
            "substances."
        ),
    },
    2: {
        "label": "Processed Culinary Ingredient",
        "description": (
            "Substances extracted from group 1 foods or from nature, such "
            "as oils, butter, sugar, and salt, used in small quantities to "
            "prepare and season food."
        ),
    },
    3: {
        "label": "Processed Food",
        "description": (
            "Group 1 foods with the addition of group 2 ingredients (oil, "
            "sugar, salt) using preservation methods such as canning or "
            "simple fermentation. Recognizable and limited ingredient list."
        ),
    },
    4: {
        "label": "Ultra-Processed Food",
        "description": (
            "Industrial formulations with five or more ingredients, "
            "typically including substances not used in home kitchens: "
            "additives, flavorings, colorings, emulsifiers, and processed "
            "protein/starch fractions."
        ),
    },
}

# ---------------------------------------------------------------------------
# Heuristic ingredient markers used for the OCR/AI fallback and for
# barcode lookups where Open Food Facts has not pre-computed a nova_group.
# This is a rule-based approximation of the NOVA methodology, not a
# substitute for the official classification.
# ---------------------------------------------------------------------------

NOVA4_STRONG_MARKERS = [
    "high fructose corn syrup", "hydrogenated oil", "hydrogenated fat",
    "partially hydrogenated", "interesterified oil", "hydrolyzed protein",
    "hydrolysed protein", "hydrolyzed vegetable protein", "maltodextrin",
    "invert sugar", "soy protein isolate", "whey protein isolate",
    "modified starch", "modified food starch", "monosodium glutamate",
    "msg", "disodium inosinate", "disodium guanylate", "artificial flavor",
    "artificial flavour", "artificial color", "artificial colour",
    "aspartame", "sucralose", "acesulfame", "saccharin", "sodium benzoate",
    "potassium sorbate", "sodium nitrite", "sodium nitrate", "bha", "bht",
    "tbhq", "polysorbate", "carrageenan", "mono- and diglycerides",
    "monoglycerides", "diglycerides", "soy lecithin", "corn syrup solids",
    "dextrose monohydrate", "glucose syrup", "caramel color",
    "caramel colour", "xanthan gum", "guar gum", "cellulose gum",
    "sodium caseinate", "calcium caseinate", "yeast extract",
    "autolyzed yeast", "textured vegetable protein", "inverted sugar syrup",
]

E_NUMBER_PATTERN = re.compile(r"\be[\s-]?\d{3,4}[a-z]?\b", re.IGNORECASE)

CULINARY_MARKERS = [
    "sugar", "salt", "vegetable oil", "olive oil", "sunflower oil",
    "butter", "honey", "vinegar", "starch",
]


def classify_nova(ingredients_text):
    """
    Heuristically classify a food product into a NOVA group (1-4) based on
    its ingredient list text.

    Returns a tuple of (nova_group: int, flagged_markers: list[str],
    confidence: str)
    """
    if not ingredients_text or not ingredients_text.strip():
        return 3, [], "low"

    text = ingredients_text.lower()

    flagged = []
    for marker in NOVA4_STRONG_MARKERS:
        if marker in text:
            flagged.append(marker)

    e_number_matches = E_NUMBER_PATTERN.findall(text)
    if e_number_matches:
        flagged.extend(sorted(set(m.strip().upper() for m in e_number_matches)))

    if flagged:
        confidence = "high" if len(flagged) >= 2 else "medium"
        return 4, sorted(set(flagged)), confidence

    ingredient_count = len([i for i in re.split(r",|;", text) if i.strip()])

    culinary_hits = [m for m in CULINARY_MARKERS if m in text]

    if ingredient_count <= 1 and not culinary_hits:
        return 1, [], "medium"

    if culinary_hits and ingredient_count <= 5:
        return 3, culinary_hits, "medium"

    if ingredient_count <= 2:
        return 2, culinary_hits, "low"

    return 3, culinary_hits, "low"


def normalize_off_product(product_json, barcode):
    """
    Normalize a raw Open Food Facts 'product' object into the shape the
    frontend expects, resolving the NOVA group from OFF data when present,
    and falling back to the heuristic classifier otherwise.
    """
    ingredients_text = (
        product_json.get("ingredients_text_en")
        or product_json.get("ingredients_text")
        or ""
    )

    off_nova_group = product_json.get("nova_group")
    flagged_markers = []
    confidence = "high"

    if isinstance(off_nova_group, int) and off_nova_group in NOVA_METADATA:
        nova_group = off_nova_group
        if nova_group == 4:
            additives = product_json.get("additives_tags", []) or []
            flagged_markers = [
                a.replace("en:", "").replace("-", " ") for a in additives
            ]
            if not flagged_markers:
                _, flagged_markers, _ = classify_nova(ingredients_text)
        confidence = "high"
    else:
        nova_group, flagged_markers, confidence = classify_nova(ingredients_text)

    nutriscore = (
        product_json.get("nutriscore_grade")
        or product_json.get("nutrition_grades")
        or None
    )
    if nutriscore:
        nutriscore = str(nutriscore).upper()

    image_url = (
        product_json.get("image_front_url")
        or product_json.get("image_url")
        or None
    )

    return {
        "found": True,
        "barcode": barcode,
        "product_name": product_json.get("product_name")
        or product_json.get("product_name_en")
        or "Unknown Product",
        "brand": product_json.get("brands", "").split(",")[0].strip()
        if product_json.get("brands")
        else None,
        "image_url": image_url,
        "ingredients_text": ingredients_text or None,
        "nova_group": nova_group,
        "nova_label": NOVA_METADATA[nova_group]["label"],
        "nova_description": NOVA_METADATA[nova_group]["description"],
        "flagged_markers": flagged_markers,
        "confidence": confidence,
        "nutriscore": nutriscore,
        "categories": product_json.get("categories", None),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/loo"""
Ultra-Processed Food (UPF) Scanner
Flask backend for Vercel serverless deployment.

Responsibilities:
1. Serve the single-page frontend.
2. Look up barcodes against the Open Food Facts API and normalize the response.
3. Classify NOVA group (1-4) either from Open Food Facts data directly, or via
   a heuristic ingredient-text classifier used as a fallback for the OCR/AI flow.
"""

import re
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

import requests

OFF_API_BASE = "https://world.openfoodfacts.org/api/v2/product"
OFF_TIMEOUT_SECONDS = 8

# ---------------------------------------------------------------------------
# NOVA classification metadata
# ---------------------------------------------------------------------------

NOVA_METADATA = {
    1: {
        "label": "Unprocessed / Minimally Processed",
        "description": (
            "Natural foods altered only by processes such as removal of "
            "inedible parts, drying, crushing, or pasteurization. No added "
            "substances."
        ),
    },
    2: {
        "label": "Processed Culinary Ingredient",
        "description": (
            "Substances extracted from group 1 foods or from nature, such "
            "as oils, butter, sugar, and salt, used in small quantities to "
            "prepare and season food."
        ),
    },
    3: {
        "label": "Processed Food",
        "description": (
            "Group 1 foods with the addition of group 2 ingredients (oil, "
            "sugar, salt) using preservation methods such as canning or "
            "simple fermentation. Recognizable and limited ingredient list."
        ),
    },
    4: {
        "label": "Ultra-Processed Food",
        "description": (
            "Industrial formulations with five or more ingredients, "
            "typically including substances not used in home kitchens: "
            "additives, flavorings, colorings, emulsifiers, and processed "
            "protein/starch fractions."
        ),
    },
}

# ---------------------------------------------------------------------------
# Heuristic ingredient markers used for the OCR/AI fallback and for
# barcode lookups where Open Food Facts has not pre-computed a nova_group.
# This is a rule-based approximation of the NOVA methodology, not a
# substitute for the official classification.
# ---------------------------------------------------------------------------

NOVA4_STRONG_MARKERS = [
    "high fructose corn syrup", "hydrogenated oil", "hydrogenated fat",
    "partially hydrogenated", "interesterified oil", "hydrolyzed protein",
    "hydrolysed protein", "hydrolyzed vegetable protein", "maltodextrin",
    "invert sugar", "soy protein isolate", "whey protein isolate",
    "modified starch", "modified food starch", "monosodium glutamate",
    "msg", "disodium inosinate", "disodium guanylate", "artificial flavor",
    "artificial flavour", "artificial color", "artificial colour",
    "aspartame", "sucralose", "acesulfame", "saccharin", "sodium benzoate",
    "potassium sorbate", "sodium nitrite", "sodium nitrate", "bha", "bht",
    "tbhq", "polysorbate", "carrageenan", "mono- and diglycerides",
    "monoglycerides", "diglycerides", "soy lecithin", "corn syrup solids",
    "dextrose monohydrate", "glucose syrup", "caramel color",
    "caramel colour", "xanthan gum", "guar gum", "cellulose gum",
    "sodium caseinate", "calcium caseinate", "yeast extract",
    "autolyzed yeast", "textured vegetable protein", "inverted sugar syrup",
]

E_NUMBER_PATTERN = re.compile(r"\be[\s-]?\d{3,4}[a-z]?\b", re.IGNORECASE)

CULINARY_MARKERS = [
    "sugar", "salt", "vegetable oil", "olive oil", "sunflower oil",
    "butter", "honey", "vinegar", "starch",
]


def classify_nova(ingredients_text):
    """
    Heuristically classify a food product into a NOVA group (1-4) based on
    its ingredient list text.

    Returns a tuple of (nova_group: int, flagged_markers: list[str],
    confidence: str)
    """
    if not ingredients_text or not ingredients_text.strip():
        return 3, [], "low"

    text = ingredients_text.lower()

    flagged = []
    for marker in NOVA4_STRONG_MARKERS:
        if marker in text:
            flagged.append(marker)

    e_number_matches = E_NUMBER_PATTERN.findall(text)
    if e_number_matches:
        flagged.extend(sorted(set(m.strip().upper() for m in e_number_matches)))

    if flagged:
        confidence = "high" if len(flagged) >= 2 else "medium"
        return 4, sorted(set(flagged)), confidence

    ingredient_count = len([i for i in re.split(r",|;", text) if i.strip()])

    culinary_hits = [m for m in CULINARY_MARKERS if m in text]

    if ingredient_count <= 1 and not culinary_hits:
        return 1, [], "medium"

    if culinary_hits and ingredient_count <= 5:
        return 3, culinary_hits, "medium"

    if ingredient_count <= 2:
        return 2, culinary_hits, "low"

    return 3, culinary_hits, "low"


def normalize_off_product(product_json, barcode):
    """
    Normalize a raw Open Food Facts 'product' object into the shape the
    frontend expects, resolving the NOVA group from OFF data when present,
    and falling back to the heuristic classifier otherwise.
    """
    ingredients_text = (
        product_json.get("ingredients_text_en")
        or product_json.get("ingredients_text")
        or ""
    )

    off_nova_group = product_json.get("nova_group")
    flagged_markers = []
    confidence = "high"

    if isinstance(off_nova_group, int) and off_nova_group in NOVA_METADATA:
        nova_group = off_nova_group
        if nova_group == 4:
            additives = product_json.get("additives_tags", []) or []
            flagged_markers = [
                a.replace("en:", "").replace("-", " ") for a in additives
            ]
            if not flagged_markers:
                _, flagged_markers, _ = classify_nova(ingredients_text)
        confidence = "high"
    else:
        nova_group, flagged_markers, confidence = classify_nova(ingredients_text)

    nutriscore = (
        product_json.get("nutriscore_grade")
        or product_json.get("nutrition_grades")
        or None
    )
    if nutriscore:
        nutriscore = str(nutriscore).upper()

    image_url = (
        product_json.get("image_front_url")
        or product_json.get("image_url")
        or None
    )

    return {
        "found": True,
        "barcode": barcode,
        "product_name": product_json.get("product_name")
        or product_json.get("product_name_en")
        or "Unknown Product",
        "brand": product_json.get("brands", "").split(",")[0].strip()
        if product_json.get("brands")
        else None,
        "image_url": image_url,
        "ingredients_text": ingredients_text or None,
        "nova_group": nova_group,
        "nova_label": NOVA_METADATA[nova_group]["label"],
        "nova_description": NOVA_METADATA[nova_group]["description"],
        "flagged_markers": flagged_markers,
        "confidence": confidence,
        "nutriscore": nutriscore,
        "categories": product_json.get("categories", None),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/lookup/<string:barcode>", methods=["GET"])
def lookup_barcode(barcode):
    barcode = barcode.strip()

    if not barcode.isdigit() or not (8 <= len(barcode) <= 14):
        return (
            jsonify(
                {
                    "found": False,
                    "error": "invalid_barcode",
                    "message": "Barcode must be 8 to 14 digits.",
                }
            ),
            400,
        )

    try:
        response = requests.get(
            f"{OFF_API_BASE}/{barcode}.json",
            timeout=OFF_TIMEOUT_SECONDS,
            headers={"User-Agent": "UPF-Scanner/1.0 (Vercel deployment)"},
        )
    except requests.exceptions.RequestException:
        return (
            jsonify(
                {
                    "found": False,
                    "error": "network_error",
                    "message": "Could not reach Open Food Facts. Please try again.",
                }
            ),
            502,
        )

    if response.status_code != 200:
        return (
            jsonify(
                {
                    "found": False,
                    "error": "upstream_error",
                    "message": "Open Food Facts returned an unexpected response.",
                }
            ),
            502,
        )

    data = response.json()

    if data.get("status") != 1 or "product" not in data:
        return (
            jsonify(
                {
                    "found": False,
                    "error": "not_found",
                    "message": "No product found for this barcode. Try the ingredient scan fallback.",
                }
            ),
            404,
        )

    normalized = normalize_off_product(data["product"], barcode)
    return jsonify(normalized), 200


@app.route("/api/analyze-ingredients", methods=["POST"])
def analyze_ingredients():
    body = request.get_json(silent=True) or {}
    ingredients_text = body.get("ingredients_text", "")

    if not isinstance(ingredients_text, str) or not ingredients_text.strip():
        return (
            jsonify(
                {
                    "error": "missing_text",
                    "message": "No ingredient text was provided to analyze.",
                }
            ),
            400,
        )

    nova_group, flagged_markers, confidence = classify_nova(ingredients_text)

    return (
        jsonify(
            {
                "found": True,
                "source": "ocr_fallback",
                "ingredients_text": ingredients_text.strip(),
                "nova_group": nova_group,
                "nova_label": NOVA_METADATA[nova_group]["label"],
                "nova_description": NOVA_METADATA[nova_group]["description"],
                "flagged_markers": flagged_markers,
                "confidence": confidence,
            }
        ),
        200,
    )


@app.errorhandler(404)
def handle_404(_error):
    return jsonify({"error": "not_found", "message": "Resource not found."}), 404


@app.errorhandler(500)
def handle_500(_error):
    return (
        jsonify({"error": "server_error", "message": "An internal error occurred."}),
        500,
    )


if __name__ == "__main__":
    app.run(debug=True)Enterkup/<string:barcode>", methods=["GET"])
def lookup_barcode(barcode):
    barcode = barcode.strip()

    if not barcode.isdigit() or not (8 <= len(barcode) <= 14):
        return (
            jsonify(
                {
                    "found": False,
                    "error": "invalid_barcode",
                    "message": "Barcode must be 8 to 14 digits.",
                }
            ),
            400,
        )

    try:
        response = requests.get(
            f"{OFF_API_BASE}/{barcode}.json",
            timeout=OFF_TIMEOUT_SECONDS,
            headers={"User-Agent": "UPF-Scanner/1.0 (Vercel deployment)"},
        )
    except requests.exceptions.RequestException:
        return (
            jsonify(
                {
                    "found": False,
                    "error": "network_error",
                    "message": "Could not reach Open Food Facts. Please try again.",
                }
            ),
            502,
        )

    if response.status_code != 200:
        return (
            jsonify(
                {
                    "found": False,
                    "error": "upstream_error",
                    "message": "Open Food Facts returned an unexpected response.",
                }
            ),
            502,
        )

    data = response.json()

    if data.get("status") != 1 or "product" not in data:
        return (
            jsonify(
                {
                    "found": False,
                    "error": "not_found",
                    "message": "No product found for this barcode. Try the ingredient scan fallback.",
                }
            ),
            404,
        )

    normalized = normalize_off_product(data["product"], barcode)
    return jsonify(normalized), 200


@app.route("/api/analyze-ingredients", methods=["POST"])
def analyze_ingredients():
    body = request.get_json(silent=True) or {}
    ingredients_text = body.get("ingredients_text", "")

    if not isinstance(ingredients_text, str) or not ingredients_text.strip():
        return (
            jsonify(
   
