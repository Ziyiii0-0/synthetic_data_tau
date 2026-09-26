#!/usr/bin/env python3
"""
Retail Database Generator

Generates a larger retail database (products, variants, users, orders) seeded
from the stock tau2 retail DB, using an LLM for product names/details and user
profiles. All data is validated against the retail policy rules.

Two-phase usage (review product names before generating the rest):
    python -m synthesis.generate_db.retail --phase 1   # writes product names
    python -m synthesis.generate_db.retail --phase 2 --output outputs/retail/db_generated.json
Without --phase both phases run back to back.
"""

import json
import random
import string
from pathlib import Path
from dataclasses import dataclass, field
import itertools

from litellm import completion

# Configuration
from synthesis.common import ASSETS_DIR, DEFAULT_MODEL, OUTPUT_DIR, TAU2_DOMAINS_DIR

SEED_DB_PATH = TAU2_DOMAINS_DIR / "retail" / "db.json"
OUTPUT_DB_PATH = OUTPUT_DIR / "retail" / "db_generated.json"
PRODUCT_NAMES_PATH = ASSETS_DIR / "retail" / "product_names.json"

# Generation targets
TARGET_PRODUCTS = 400
VARIANTS_PER_PRODUCT_MIN = 5
VARIANTS_PER_PRODUCT_MAX = 20  # Match original dataset distribution (mean ~11)
TARGET_ORDERS = 2000  # Scaled proportionally
TARGET_USERS = 500   # Scaled proportionally


# Product categories for diversity
PRODUCT_CATEGORIES = [
    "Electronics & Technology",
    "Home & Kitchen",
    "Sports & Outdoors", 
    "Fashion & Apparel",
    "Health & Beauty",
    "Toys & Games",
    "Office & School Supplies",
    "Automotive & Tools",
    "Garden & Patio",
    "Pet Supplies",
    "Baby & Kids",
    "Books & Media",
    "Food & Beverages",
    "Arts & Crafts",
    "Musical Instruments",
    "Furniture & Decor",
    "Fitness & Wellness",
    "Travel & Luggage",
    "Jewelry & Watches",
    "Hobbies & Collections"
]

# US States with cities and zip code prefixes
US_LOCATIONS = {
    "CA": {"cities": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Sacramento"], "zip_prefix": ["90", "91", "92", "94", "95"]},
    "NY": {"cities": ["New York", "Brooklyn", "Queens", "Buffalo", "Rochester"], "zip_prefix": ["10", "11", "12", "14"]},
    "TX": {"cities": ["Houston", "Dallas", "Austin", "San Antonio", "Fort Worth"], "zip_prefix": ["75", "76", "77", "78"]},
    "FL": {"cities": ["Miami", "Orlando", "Tampa", "Jacksonville", "Fort Lauderdale"], "zip_prefix": ["32", "33", "34"]},
    "IL": {"cities": ["Chicago", "Aurora", "Naperville", "Rockford", "Joliet"], "zip_prefix": ["60", "61", "62"]},
    "PA": {"cities": ["Philadelphia", "Pittsburgh", "Allentown", "Erie", "Reading"], "zip_prefix": ["15", "16", "17", "18", "19"]},
    "AZ": {"cities": ["Phoenix", "Tucson", "Mesa", "Scottsdale", "Chandler"], "zip_prefix": ["85", "86"]},
    "CO": {"cities": ["Denver", "Colorado Springs", "Aurora", "Boulder", "Fort Collins"], "zip_prefix": ["80", "81"]},
    "WA": {"cities": ["Seattle", "Spokane", "Tacoma", "Vancouver", "Bellevue"], "zip_prefix": ["98", "99"]},
    "MA": {"cities": ["Boston", "Cambridge", "Worcester", "Springfield", "Lowell"], "zip_prefix": ["01", "02"]},
}

# Street types for address generation
STREET_TYPES = ["Street", "Avenue", "Boulevard", "Drive", "Lane", "Road", "Way", "Place", "Court"]
STREET_NAMES = ["Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Washington", "Lincoln", "Park", 
                "Lake", "Hill", "River", "Sunset", "Spring", "Valley", "Forest", "Meadow", "Highland"]

# Credit card brands
CC_BRANDS = ["visa", "mastercard", "amex", "discover"]

# Order statuses and their distribution
ORDER_STATUS_WEIGHTS = {
    "pending": 0.15,
    "processed": 0.35,
    "delivered": 0.40,
    "cancelled": 0.10,
}


@dataclass
class GeneratedDatabase:
    """Container for the generated database"""
    products: dict = field(default_factory=dict)
    users: dict = field(default_factory=dict)
    orders: dict = field(default_factory=dict)


def load_seed_data() -> dict:
    """Load the existing seed database"""
    with open(SEED_DB_PATH, "r") as f:
        return json.load(f)


def generate_unique_id(length: int = 10, existing_ids: set = None) -> str:
    """Generate a unique numeric ID"""
    existing_ids = existing_ids or set()
    while True:
        new_id = "".join(random.choices(string.digits, k=length))
        if new_id not in existing_ids and new_id[0] != "0":
            return new_id


def generate_order_id(existing_ids: set = None) -> str:
    """Generate a unique order ID in format #W1234567"""
    existing_ids = existing_ids or set()
    while True:
        order_id = f"#W{''.join(random.choices(string.digits, k=7))}"
        if order_id not in existing_ids:
            return order_id


def generate_tracking_id() -> str:
    """Generate a 12-digit tracking ID"""
    return "".join(random.choices(string.digits, k=12))


def generate_address() -> dict:
    """Generate a realistic US address"""
    state = random.choice(list(US_LOCATIONS.keys()))
    location = US_LOCATIONS[state]
    city = random.choice(location["cities"])
    zip_prefix = random.choice(location["zip_prefix"])
    
    street_num = random.randint(100, 999)
    street_name = random.choice(STREET_NAMES)
    street_type = random.choice(STREET_TYPES)
    suite_num = random.randint(100, 999)
    
    return {
        "address1": f"{street_num} {street_name} {street_type}",
        "address2": f"Suite {suite_num}",
        "city": city,
        "country": "USA",
        "state": state,
        "zip": f"{zip_prefix}{random.randint(100, 999):03d}"
    }


def generate_payment_method_id(source: str) -> str:
    """Generate a payment method ID"""
    return f"{source}_{random.randint(1000000, 9999999)}"


def generate_payment_methods() -> dict:
    """Generate 1-3 payment methods for a user"""
    methods = {}
    num_methods = random.randint(1, 3)
    sources = random.sample(["gift_card", "credit_card", "paypal"], num_methods)
    
    for source in sources:
        method_id = generate_payment_method_id(source)
        if source == "gift_card":
            methods[method_id] = {
                "source": "gift_card",
                "id": method_id,
                "balance": round(random.uniform(10, 500), 2)
            }
        elif source == "credit_card":
            methods[method_id] = {
                "source": "credit_card",
                "id": method_id,
                "brand": random.choice(CC_BRANDS),
                "last_four": "".join(random.choices(string.digits, k=4))
            }
        else:  # paypal
            methods[method_id] = {
                "source": "paypal",
                "id": method_id
            }
    
    return methods


def is_similar_name(new_name: str, existing_names: set) -> bool:
    """Check if a name is too similar to existing names"""
    new_words = set(new_name.lower().split())
    # Remove common words
    common_words = {"with", "and", "for", "the", "a", "an", "of", "in", "on"}
    new_words = new_words - common_words
    
    for existing in existing_names:
        existing_words = set(existing.lower().split()) - common_words
        # If main noun is the same, consider it duplicate
        if new_words & existing_words:
            overlap = len(new_words & existing_words)
            if overlap >= 1 and len(new_words) <= 3:
                return True
    return False


def generate_product_names_only(
    seed_products: dict,
    target_count: int,
    model: str = DEFAULT_MODEL
) -> list[dict]:
    """Phase 1: Generate diverse product names by category"""
    
    existing_names = {p["name"] for p in seed_products.values()}
    all_names = []
    
    # Add seed product names with their categories
    for p in seed_products.values():
        all_names.append({"name": p["name"], "category": "Existing (seed)"})
    
    # Get seed product names as examples of good short names
    seed_examples = [p["name"] for p in list(seed_products.values())[:15]]
    
    new_names_needed = target_count - len(all_names)
    names_per_category = new_names_needed // len(PRODUCT_CATEGORIES) + 1
    
    print(f"Generating {new_names_needed} new product names across {len(PRODUCT_CATEGORIES)} categories...")
    print(f"  (~{names_per_category} products per category)")
    
    for category in PRODUCT_CATEGORIES:
        batch_size = min(names_per_category, 50)
        names_in_category = 0
        max_batches = 3  # Max batches per category
        
        for batch_attempt in range(max_batches):
            if names_in_category >= names_per_category:
                break
                
            count_needed = min(batch_size, names_per_category - names_in_category)
            
            prompt = f"""Generate {count_needed} COMMON everyday product names for: "{category}"

RULES:
1. SHORT names (1-3 words), like: {json.dumps(seed_examples[:8])}

2. COMMON products that people actually buy often (no niche/weird items)
   GOOD: "Toaster", "Pillow", "Umbrella", "Wallet"
   BAD: "Smart Pet Feeder", "UV Sanitizer Wand", "Aromatherapy Diffuser"

3. Do not add a lot of adjectives or modifiers - just the core product name
   BAD: "Wireless Bluetooth Speaker", "Portable Mini Fan"
   GOOD: "Speaker", "Mini Fan", "Blender"

Already used (skip these):
{json.dumps(list(existing_names)[-30:], indent=2)}

Return JSON array: ["Product1", "Product2", ...]"""

            try:
                messages = [{"role": "user", "content": prompt}]
                response = completion(
                    model=model,
                    messages=messages,
                    temperature=0.9,
                    max_tokens=2000,
                    num_retries=2
                )
                
                # Check for empty response
                if not response.choices or not response.choices[0].message.content:
                    print(f"    Warning: Empty response for {category}")
                    continue
                
                response_text = response.choices[0].message.content.strip()
                
                # Handle markdown code blocks
                if "```" in response_text:
                    # Extract content between ``` markers
                    parts = response_text.split("```")
                    for part in parts:
                        if part.strip().startswith("json"):
                            response_text = part.strip()[4:].strip()
                            break
                        elif part.strip().startswith("["):
                            response_text = part.strip()
                            break
                
                # Extract JSON array from response (LLM may add text before/after)
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx + 1]
                
                # Skip if still empty or no array found
                if not response_text or not response_text.startswith("["):
                    print(f"    Warning: No JSON array found for {category}")
                    continue
                
                new_names = json.loads(response_text)
                
                for name in new_names:
                    # Skip if too long (more than 4 words)
                    if len(name.split()) > 4:
                        continue
                    # Skip if duplicate or too similar
                    if name in existing_names:
                        continue
                    if is_similar_name(name, existing_names):
                        continue
                    if len(all_names) < target_count:
                        existing_names.add(name)
                        all_names.append({"name": name, "category": category})
                        names_in_category += 1
                
            except json.JSONDecodeError as e:
                print(f"    JSON Error in {category}: {e}")
                print(f"    Raw response: {response_text[:200] if response_text else 'EMPTY'}...")
                continue
            except Exception as e:
                print(f"    Error in {category}: {type(e).__name__}: {e}")
                continue
        
        print(f"  {category}: {names_in_category} products (total: {len(all_names)}/{target_count})")
    
    return all_names


def save_product_names(names: list[dict], output_path: Path):
    """Save product names for review"""
    # Group by category for easier review
    by_category = {}
    for item in names:
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item["name"])
    
    output = {
        "total_count": len(names),
        "by_category": by_category,
        "all_names": names
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nProduct names saved to {output_path}")
    print(f"Total: {len(names)} products across {len(by_category)} categories")
    print("\nCategory breakdown:")
    for cat, prods in sorted(by_category.items()):
        print(f"  {cat}: {len(prods)} products")


def load_product_names(input_path: Path) -> list[dict]:
    """Load approved product names from by_category (skip seed products)"""
    with open(input_path, "r") as f:
        data = json.load(f)
    
    all_names = []
    by_category = data.get("by_category", {})
    
    for category, names in by_category.items():
        # Skip seed products
        if category == "Existing (seed)":
            continue
        for name in names:
            all_names.append({"name": name, "category": category})
    
    return all_names


def generate_product_details(
    product_names: list[dict],
    seed_products: dict,
    model: str = DEFAULT_MODEL
) -> dict:
    """Phase 2: Generate product details (options, prices) for approved names
    
    Note: seed_products is only used as reference examples, NOT included in output.
    """
    
    products = {}  # Start fresh, don't include seed products
    all_product_ids = set()
    
    # Use all provided names (they should already exclude seed products)
    new_names = product_names
    
    print(f"Generating details for {len(new_names)} products (seed data used as reference only)...")
    
    # Get example products for reference
    example_products = []
    for p in list(seed_products.values())[:5]:
        variants = list(p["variants"].values())
        if variants:
            example_products.append({
                "name": p["name"],
                "options": list(variants[0]["options"].keys()),
                "base_price": round(sum(v["price"] for v in variants) / len(variants), 2)
            })
    
    # Process in batches
    batch_size = 15
    for i in range(0, len(new_names), batch_size):
        batch = new_names[i:i + batch_size]
        batch_names = [n["name"] for n in batch]
        
        prompt = f"""For each of these products, generate the product options and pricing.

Products to define:
{json.dumps(batch_names, indent=2)}

Example format from our catalog:
{json.dumps(example_products, indent=2)}

For each product, return:
- "name": The exact product name from the list above
- "options": Object with 2-4 option types, each with 3-5 values
- "base_price": Realistic base price (number)
- "price_variance": How much price can vary (0.1 = 10%)

Return a JSON array:
[
  {{
    "name": "Product Name",
    "options": {{"color": ["red", "blue"], "size": ["S", "M", "L"]}},
    "base_price": 49.99,
    "price_variance": 0.15
  }}
]

Return ONLY the JSON array."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = completion(
                model=model,
                messages=messages,
                temperature=0.5,
                max_tokens=4000,
                num_retries=3
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            product_defs = json.loads(response_text)
            
            for product_def in product_defs:
                product_id = generate_unique_id(10, all_product_ids)
                all_product_ids.add(product_id)
                
                products[product_id] = {
                    "name": product_def["name"],
                    "product_id": product_id,
                    "variants": {},
                    "_options": product_def.get("options", {}),
                    "_base_price": product_def.get("base_price", 50),
                    "_price_variance": product_def.get("price_variance", 0.15)
                }
            
            print(f"  Processed {min(i + batch_size, len(new_names))}/{len(new_names)} products")
            
        except Exception as e:
            print(f"  Error processing batch {i // batch_size + 1}: {e}")
            # Add products with default options as fallback
            for name_item in batch:
                product_id = generate_unique_id(10, all_product_ids)
                all_product_ids.add(product_id)
                products[product_id] = {
                    "name": name_item["name"],
                    "product_id": product_id,
                    "variants": {},
                    "_options": {"type": ["standard", "premium"], "color": ["black", "white", "gray"]},
                    "_base_price": 50,
                    "_price_variance": 0.15
                }
    
    return products


def generate_variants_for_product(
    product: dict,
    all_item_ids: set
) -> dict:
    """Generate 5-10 variants for a product"""
    
    variants = {}
    
    # Check if product already has variants (seed products)
    if product.get("variants") and len(product["variants"]) >= VARIANTS_PER_PRODUCT_MIN:
        return product["variants"]
    
    # Get options from the product definition
    options = product.get("_options", {})
    base_price = product.get("_base_price", 50)
    price_variance = product.get("_price_variance", 0.15)
    
    if not options:
        # Fallback for seed products without _options
        if product.get("variants"):
            sample_variant = list(product["variants"].values())[0]
            options = {k: [v] for k, v in sample_variant["options"].items()}
            base_price = sample_variant["price"]
    
    if not options:
        return variants
    
    # Generate all possible combinations
    option_keys = list(options.keys())
    option_values = [options[k] for k in option_keys]
    all_combinations = list(itertools.product(*option_values))
    
    # Use normal distribution centered at 11 (like original dataset)
    # Clamp between MIN and MAX
    target_variants = int(random.gauss(11, 4))  # mean=11, std=4
    target_variants = max(VARIANTS_PER_PRODUCT_MIN, min(VARIANTS_PER_PRODUCT_MAX, target_variants))
    num_variants = min(target_variants, len(all_combinations))
    selected_combinations = random.sample(all_combinations, num_variants)
    
    for combo in selected_combinations:
        item_id = generate_unique_id(10, all_item_ids)
        all_item_ids.add(item_id)
        
        # Calculate price with some variance
        price_multiplier = 1 + random.uniform(-price_variance, price_variance)
        price = round(base_price * price_multiplier, 2)
        
        variant_options = {option_keys[i]: combo[i] for i in range(len(option_keys))}
        
        variants[item_id] = {
            "item_id": item_id,
            "options": variant_options,
            "available": random.random() > 0.2,  # 80% availability
            "price": price
        }
    
    return variants


def generate_all_variants(products: dict) -> tuple[dict, set]:
    """Generate variants for all products"""
    all_item_ids = set()
    
    # First, collect existing item IDs
    for product in products.values():
        for item_id in product.get("variants", {}).keys():
            all_item_ids.add(item_id)
    
    print(f"Generating variants for {len(products)} products...")
    
    for i, (product_id, product) in enumerate(products.items()):
        variants = generate_variants_for_product(product, all_item_ids)
        products[product_id]["variants"] = variants
        
        # Clean up temporary fields
        products[product_id].pop("_options", None)
        products[product_id].pop("_base_price", None)
        products[product_id].pop("_price_variance", None)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(products)} products")
    
    return products, all_item_ids


def generate_users_with_llm(
    target_count: int,
    existing_users: dict = None,
    model: str = DEFAULT_MODEL
) -> dict:
    """Generate users with LLM-generated names"""
    
    users = dict(existing_users) if existing_users else {}
    existing_user_ids = set(users.keys())
    existing_emails = {u["email"] for u in users.values()}
    
    new_users_needed = target_count - len(users)
    print(f"Generating {new_users_needed} new users...")
    
    # Generate names in batches
    batch_size = 100
    all_names = []
    
    for batch_start in range(0, new_users_needed, batch_size):
        batch_count = min(batch_size, new_users_needed - batch_start)
        
        prompt = f"""Generate exactly {batch_count} unique realistic American first and last name pairs. 
Include diverse names representing various ethnic backgrounds common in the USA.

Return as a JSON array of objects with "first_name" and "last_name" fields.
Example: [{{"first_name": "Michael", "last_name": "Johnson"}}, {{"first_name": "Priya", "last_name": "Patel"}}]

Return ONLY the JSON array, no other text."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = completion(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=4000,
                num_retries=3
            )
            
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            names_batch = json.loads(response_text)
            all_names.extend(names_batch)
            print(f"  Generated names batch {batch_start // batch_size + 1}")
            
        except Exception as e:
            print(f"  Error generating names: {e}")
            # Fallback names
            fallback_first = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "James", "Emma"]
            fallback_last = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
            for _ in range(batch_count):
                all_names.append({
                    "first_name": random.choice(fallback_first),
                    "last_name": random.choice(fallback_last)
                })
    
    # Create user records
    for name_data in all_names:
        first_name = name_data["first_name"]
        last_name = name_data["last_name"]
        
        # Generate unique user_id
        while True:
            suffix = random.randint(1000, 9999)
            user_id = f"{first_name.lower()}_{last_name.lower()}_{suffix}"
            if user_id not in existing_user_ids:
                break
        
        existing_user_ids.add(user_id)
        
        # Generate unique email
        while True:
            email_suffix = random.randint(1000, 9999)
            email = f"{first_name.lower()}.{last_name.lower()}{email_suffix}@example.com"
            if email not in existing_emails:
                break
        
        existing_emails.add(email)
        
        users[user_id] = {
            "user_id": user_id,
            "name": {
                "first_name": first_name,
                "last_name": last_name
            },
            "address": generate_address(),
            "email": email,
            "payment_methods": generate_payment_methods(),
            "orders": []
        }
    
    print(f"  Total users: {len(users)}")
    return users


def generate_orders(
    products: dict,
    users: dict,
    target_count: int,
    existing_orders: dict = None
) -> dict:
    """Generate orders with proper policy compliance"""
    
    orders = dict(existing_orders) if existing_orders else {}
    existing_order_ids = set(orders.keys())
    
    # Build item lookup for all items (order was placed when item was available)
    all_items = []
    for product in products.values():
        for item_id, variant in product.get("variants", {}).items():
            all_items.append({
                "name": product["name"],
                "product_id": product["product_id"],
                "item_id": item_id,
                "price": variant["price"],
                "options": variant["options"]
            })
    
    if not all_items:
        print("Warning: No items found!")
        return orders
    
    new_orders_needed = target_count - len(orders)
    print(f"Generating {new_orders_needed} new orders...")
    
    user_list = list(users.keys())
    
    for i in range(new_orders_needed):
        order_id = generate_order_id(existing_order_ids)
        existing_order_ids.add(order_id)
        
        # Select random user
        user_id = random.choice(user_list)
        user = users[user_id]
        
        # Select 1-5 items for this order
        num_items = random.randint(1, 5)
        order_items = random.sample(all_items, min(num_items, len(all_items)))
        
        # Calculate total
        total = sum(item["price"] for item in order_items)
        
        # Select status based on weights
        status = random.choices(
            list(ORDER_STATUS_WEIGHTS.keys()),
            weights=list(ORDER_STATUS_WEIGHTS.values())
        )[0]
        
        # Select payment method from user's methods
        payment_methods = list(user["payment_methods"].keys())
        payment_method_id = random.choice(payment_methods)
        
        # Build payment history
        payment_history = [{
            "transaction_type": "payment",
            "amount": round(total, 2),
            "payment_method_id": payment_method_id
        }]
        
        # Build fulfillments
        fulfillments = []
        if status in ["processed", "delivered", "cancelled"]:
            fulfillments = [{
                "tracking_id": [generate_tracking_id()],
                "item_ids": [item["item_id"] for item in order_items]
            }]
        
        # Add refund for cancelled orders
        if status == "cancelled":
            payment_history.append({
                "transaction_type": "refund",
                "amount": round(total, 2),
                "payment_method_id": payment_method_id
            })
        
        order = {
            "order_id": order_id,
            "user_id": user_id,
            "address": user["address"].copy(),
            "items": order_items,
            "status": status,
            "fulfillments": fulfillments,
            "payment_history": payment_history
        }
        
        orders[order_id] = order
        
        # Add order to user's order list
        if order_id not in user["orders"]:
            user["orders"].append(order_id)
        
        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{new_orders_needed} orders")
    
    return orders


def validate_database(db: GeneratedDatabase) -> list[str]:
    """Validate the generated database for policy compliance"""
    
    errors = []
    
    print("Validating database...")
    
    # Check product structure
    product_ids = set()
    all_item_ids = {}  # item_id -> product info
    
    for product_id, product in db.products.items():
        if product_id in product_ids:
            errors.append(f"Duplicate product_id: {product_id}")
        product_ids.add(product_id)
        
        if not product.get("variants"):
            errors.append(f"Product {product_id} has no variants")
        else:
            for item_id, variant in product["variants"].items():
                if item_id in all_item_ids:
                    errors.append(f"Duplicate item_id: {item_id}")
                all_item_ids[item_id] = {
                    "product_id": product_id,
                    "name": product["name"],
                    "options": variant["options"],
                    "price": variant["price"]
                }
    
    print(f"  Products: {len(product_ids)}, Items: {len(all_item_ids)}")
    
    # Check user structure
    user_ids = set()
    all_payment_methods = {}  # payment_method_id -> user_id
    
    for user_id, user in db.users.items():
        if user_id in user_ids:
            errors.append(f"Duplicate user_id: {user_id}")
        user_ids.add(user_id)
        
        for pm_id, pm in user.get("payment_methods", {}).items():
            all_payment_methods[pm_id] = user_id
            
            if pm["source"] == "gift_card" and pm.get("balance", 0) < 0:
                errors.append(f"Negative gift card balance for {pm_id}")
    
    print(f"  Users: {len(user_ids)}, Payment methods: {len(all_payment_methods)}")
    
    # Check order structure
    order_ids = set()
    user_order_map = {uid: set() for uid in user_ids}
    
    for order_id, order in db.orders.items():
        if order_id in order_ids:
            errors.append(f"Duplicate order_id: {order_id}")
        order_ids.add(order_id)
        
        # Check user exists
        if order["user_id"] not in user_ids:
            errors.append(f"Order {order_id} references non-existent user: {order['user_id']}")
        else:
            user_order_map[order["user_id"]].add(order_id)
        
        # Check items exist
        for item in order.get("items", []):
            if item["item_id"] not in all_item_ids:
                errors.append(f"Order {order_id} references non-existent item: {item['item_id']}")
        
        # Check payment method exists
        for payment in order.get("payment_history", []):
            pm_id = payment.get("payment_method_id")
            if pm_id and pm_id not in all_payment_methods:
                errors.append(f"Order {order_id} references non-existent payment method: {pm_id}")
        
        # Check cancelled orders have refunds
        if order["status"] == "cancelled":
            has_refund = any(p["transaction_type"] == "refund" for p in order.get("payment_history", []))
            if not has_refund:
                errors.append(f"Cancelled order {order_id} has no refund")
        
        # Check pending orders have empty fulfillments
        if order["status"] == "pending" and order.get("fulfillments"):
            errors.append(f"Pending order {order_id} should have empty fulfillments")
    
    print(f"  Orders: {len(order_ids)}")
    
    # Check user.orders matches actual orders
    for user_id, user in db.users.items():
        user_orders_set = set(user.get("orders", []))
        actual_orders = user_order_map.get(user_id, set())
        
        if user_orders_set != actual_orders:
            missing = actual_orders - user_orders_set
            extra = user_orders_set - actual_orders
            if missing:
                errors.append(f"User {user_id} missing orders in list: {missing}")
            if extra:
                errors.append(f"User {user_id} has extra orders in list: {extra}")
    
    if errors:
        print(f"  Found {len(errors)} validation errors")
    else:
        print("  Validation passed!")
    
    return errors


def fix_user_order_lists(users: dict, orders: dict) -> dict:
    """Ensure user.orders lists match the orders in the orders table"""
    
    # Build correct mapping
    user_order_map = {uid: [] for uid in users.keys()}
    for order_id, order in orders.items():
        if order["user_id"] in user_order_map:
            user_order_map[order["user_id"]].append(order_id)
    
    # Update users
    for user_id in users:
        users[user_id]["orders"] = user_order_map.get(user_id, [])
    
    return users


def save_database(db: GeneratedDatabase, output_path: Path):
    """Save the generated database to JSON"""
    
    output = {
        "products": db.products,
        "users": db.users,
        "orders": db.orders
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    
    print(f"Database saved to {output_path}")


def main():
    """Main generation pipeline with two-phase support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Retail Database Generator")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                       help="Phase 1: Generate product names only. Phase 2: Generate full database from approved names.")
    parser.add_argument("--names-file", type=str, default=str(PRODUCT_NAMES_PATH),
                       help="Path to product names JSON file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DB_PATH),
                       help=f"Output database path (default: {OUTPUT_DB_PATH})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="LiteLLM model used for product and user generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    random.seed(args.seed)
    
    print("=" * 60)
    print("Retail Database Generator")
    print("=" * 60)
    
    # Load seed data
    print("\n[1] Loading seed data...")
    seed_data = load_seed_data()
    print(f"  Loaded {len(seed_data['products'])} seed products")
    
    if args.phase == 1:
        # Phase 1: Generate product names only
        print("\n" + "=" * 60)
        print("PHASE 1: Generating Product Names")
        print("=" * 60)
        
        print("\n[2] Generating diverse product names...")
        product_names = generate_product_names_only(
            seed_data["products"], TARGET_PRODUCTS, model=args.model
        )
        
        # Save for review
        save_product_names(product_names, Path(args.names_file))
        
        print("\n" + "=" * 60)
        print("Phase 1 Complete!")
        print("=" * 60)
        print(f"\nPlease review the product names in: {args.names_file}")
        print("After review, run with --phase 2 to generate full database.")
        return
    
    elif args.phase == 2:
        # Phase 2: Generate full database from approved names
        print("\n" + "=" * 60)
        print("PHASE 2: Generating Full Database")
        print("=" * 60)
        
        names_path = Path(args.names_file)
        if not names_path.exists():
            print(f"Error: Names file not found: {names_path}")
            print("Please run --phase 1 first to generate product names.")
            return
        
        print(f"\n[2] Loading approved product names from {names_path}...")
        product_names = load_product_names(names_path)
        print(f"  Loaded {len(product_names)} product names")
        
        print("\n[3] Generating product details...")
        products = generate_product_details(product_names, seed_data["products"], model=args.model)
        
    else:
        # Default: Run both phases
        print("\nRunning full generation (both phases)...")
        
        print("\n[2] Generating product names...")
        product_names = generate_product_names_only(
            seed_data["products"], TARGET_PRODUCTS, model=args.model
        )
        
        print("\n[3] Generating product details...")
        products = generate_product_details(product_names, seed_data["products"], model=args.model)
    
    # Initialize database
    db = GeneratedDatabase()
    db.products = products
    
    # Generate variants
    print("\n[4] Generating variants...")
    db.products, all_item_ids = generate_all_variants(db.products)
    
    # Generate users (fresh, no seed data)
    print("\n[5] Generating users with LLM...")
    db.users = generate_users_with_llm(
        TARGET_USERS, existing_users=None, model=args.model  # Don't include seed users
    )
    
    # Generate orders (fresh, no seed data)
    print("\n[6] Generating orders...")
    db.orders = generate_orders(
        db.products, db.users, TARGET_ORDERS, existing_orders=None  # Don't include seed orders
    )
    
    # Fix user order lists
    print("\n[7] Fixing user-order associations...")
    db.users = fix_user_order_lists(db.users, db.orders)
    
    # Validate
    print("\n[8] Validating database...")
    errors = validate_database(db)
    
    if errors:
        print("\nValidation errors found:")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    
    # Save
    print("\nSaving database...")
    save_database(db, Path(args.output))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Products: {len(db.products)}")
    total_variants = sum(len(p.get('variants', {})) for p in db.products.values())
    print(f"Total variants: {total_variants}")
    print(f"Users: {len(db.users)}")
    print(f"Orders: {len(db.orders)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

