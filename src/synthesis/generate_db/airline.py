#!/usr/bin/env python3
"""
Airline Database Generator

Generates a scaled airline database with flights, users, and reservations
using LLM API for realistic user name generation. Ensures all data conforms
to airline policy rules.
"""

import json
import random
import string
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from litellm import completion

from synthesis.common import DEFAULT_MODEL, OUTPUT_DIR, TAU2_DOMAINS_DIR

SEED_DB_PATH = TAU2_DOMAINS_DIR / "airline" / "db.json"
OUTPUT_DB_PATH = OUTPUT_DIR / "airline" / "db_generated.json"

# Generation targets
TARGET_FLIGHTS = 400
TARGET_USERS = 500
TARGET_RESERVATIONS = 2000


# The current time in the domain is 2024-05-15 15:00:00 EST
CURRENT_DATE = datetime(2024, 5, 15)
DATE_RANGE_START = datetime(2024, 5, 1)
DATE_RANGE_END = datetime(2024, 5, 30)

# Airports (20 major US airports matching existing DB)
AIRPORTS = [
    "ATL", "BOS", "CLT", "DEN", "DFW", "DTW", "EWR", "IAH",
    "JFK", "LAS", "LAX", "LGA", "MCO", "MIA", "MSP", "ORD",
    "PHL", "PHX", "SEA", "SFO"
]

# Cabin classes and pricing multipliers
CABIN_CLASSES = ["basic_economy", "economy", "business"]
CABIN_PRICE_MULTIPLIER = {
    "basic_economy": 0.7,
    "economy": 1.0,
    "business": 3.5,
}

# Membership levels and their distribution
MEMBERSHIP_LEVELS = ["regular", "silver", "gold"]
MEMBERSHIP_WEIGHTS = [0.30, 0.35, 0.35]

# Flight status distribution for past dates
PAST_STATUS_WEIGHTS = {
    "landed": 0.85,
    "cancelled": 0.08,
    "delayed": 0.04,
    "flying": 0.03,
}

# Reservation distribution
INSURANCE_RATE = 0.50
CABIN_WEIGHTS = {"basic_economy": 0.33, "economy": 0.34, "business": 0.33}
TRIP_TYPE_WEIGHTS = {"one_way": 0.52, "round_trip": 0.48}

# Baggage rules per membership and cabin
FREE_BAGS = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 4},
}

# US locations for addresses
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

STREET_TYPES = ["Street", "Avenue", "Boulevard", "Drive", "Lane", "Road", "Way", "Place", "Court"]
STREET_NAMES = ["Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Washington", "Lincoln", "Park",
                "Lake", "Hill", "River", "Sunset", "Spring", "Valley", "Forest", "Meadow", "Highland"]

CC_BRANDS = ["visa", "mastercard", "amex", "discover"]


@dataclass
class GeneratedDatabase:
    """Container for the generated database"""
    flights: dict = field(default_factory=dict)
    users: dict = field(default_factory=dict)
    reservations: dict = field(default_factory=dict)


def generate_flight_number(existing: set) -> str:
    """Generate a unique flight number in HAT### format"""
    while True:
        num = random.randint(1, 999)
        flight_num = f"HAT{num:03d}"
        if flight_num not in existing:
            return flight_num


def generate_reservation_id(existing: set) -> str:
    """Generate a unique 6-character alphanumeric reservation ID"""
    while True:
        res_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if res_id not in existing:
            return res_id


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


def generate_payment_methods() -> dict:
    """Generate 2-4 payment methods for a user (credit card, gift card, certificate)"""
    methods = {}
    # Always have at least one credit card
    num_credit_cards = random.randint(1, 2)
    num_gift_cards = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
    num_certificates = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]

    for _ in range(num_credit_cards):
        method_id = f"credit_card_{random.randint(1000000, 9999999)}"
        methods[method_id] = {
            "source": "credit_card",
            "id": method_id,
            "brand": random.choice(CC_BRANDS),
            "last_four": "".join(random.choices(string.digits, k=4))
        }

    for _ in range(num_gift_cards):
        method_id = f"gift_card_{random.randint(1000000, 9999999)}"
        methods[method_id] = {
            "source": "gift_card",
            "id": method_id,
            "amount": random.choice([50, 100, 150, 200, 250, 300, 500])
        }

    for _ in range(num_certificates):
        method_id = f"certificate_{random.randint(1000000, 9999999)}"
        methods[method_id] = {
            "source": "certificate",
            "id": method_id,
            "amount": random.choice([50, 100, 150, 200, 250, 500])
        }

    return methods


def generate_dob() -> str:
    """Generate a realistic date of birth (ages 20-75)"""
    year = random.randint(1949, 2004)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def generate_flights(target_count: int) -> dict:
    """Generate flight routes with date-based availability"""
    flights = {}
    existing_numbers = set()

    # Generate routes: pick origin-destination pairs
    # Each flight covers a specific route
    routes = []
    for origin in AIRPORTS:
        for dest in AIRPORTS:
            if origin != dest:
                routes.append((origin, dest))

    # Sample routes for our flights
    selected_routes = random.choices(routes, k=target_count)

    # Generate departure/arrival times (realistic)
    for i, (origin, dest) in enumerate(selected_routes):
        flight_number = generate_flight_number(existing_numbers)
        existing_numbers.add(flight_number)

        # Generate scheduled times
        dep_hour = random.randint(5, 22)
        dep_minute = random.choice([0, 15, 30, 45])
        # Flight duration: 1-6 hours
        duration_hours = random.randint(1, 5)
        duration_minutes = random.choice([0, 15, 30, 45])

        arr_hour = dep_hour + duration_hours
        arr_minute = dep_minute + duration_minutes
        if arr_minute >= 60:
            arr_hour += 1
            arr_minute -= 60
        arr_hour = min(arr_hour, 23)

        scheduled_departure = f"{dep_hour:02d}:{dep_minute:02d}:00"
        scheduled_arrival = f"{arr_hour:02d}:{arr_minute:02d}:00"

        # Generate dates
        dates = {}
        current = DATE_RANGE_START
        while current <= DATE_RANGE_END:
            date_str = current.strftime("%Y-%m-%d")

            if current < CURRENT_DATE:
                # Past date
                status = random.choices(
                    list(PAST_STATUS_WEIGHTS.keys()),
                    weights=list(PAST_STATUS_WEIGHTS.values())
                )[0]

                if status == "landed":
                    dep_var = random.randint(-30, 30)
                    arr_var = random.randint(-30, 30)
                    actual_dep = datetime(current.year, current.month, current.day,
                                         dep_hour, dep_minute) + timedelta(minutes=dep_var)
                    actual_arr = datetime(current.year, current.month, current.day,
                                         arr_hour, arr_minute) + timedelta(minutes=arr_var)
                    dates[date_str] = {
                        "status": "landed",
                        "actual_departure_time_est": actual_dep.strftime("%Y-%m-%dT%H:%M:%S"),
                        "actual_arrival_time_est": actual_arr.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                elif status == "cancelled":
                    dates[date_str] = {"status": "cancelled"}
                elif status == "delayed":
                    delay_min = random.randint(15, 120)
                    est_dep = datetime(current.year, current.month, current.day,
                                       dep_hour, dep_minute) + timedelta(minutes=delay_min)
                    est_arr = datetime(current.year, current.month, current.day,
                                       arr_hour, arr_minute) + timedelta(minutes=delay_min)
                    dates[date_str] = {
                        "status": "delayed",
                        "estimated_departure_time_est": est_dep.strftime("%Y-%m-%dT%H:%M:%S"),
                        "estimated_arrival_time_est": est_arr.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                else:  # flying
                    dep_var = random.randint(-15, 15)
                    actual_dep = datetime(current.year, current.month, current.day,
                                         dep_hour, dep_minute) + timedelta(minutes=dep_var)
                    est_arr = datetime(current.year, current.month, current.day,
                                       arr_hour, arr_minute) + timedelta(minutes=random.randint(-10, 30))
                    dates[date_str] = {
                        "status": "flying",
                        "actual_departure_time_est": actual_dep.strftime("%Y-%m-%dT%H:%M:%S"),
                        "estimated_arrival_time_est": est_arr.strftime("%Y-%m-%dT%H:%M:%S")
                    }

            elif current.date() == CURRENT_DATE.date():
                # Current date - mix of statuses
                status = random.choices(
                    ["on time", "delayed", "flying", "landed", "available"],
                    weights=[0.2, 0.1, 0.1, 0.3, 0.3]
                )[0]
                if status == "available":
                    base_price = random.randint(80, 400)
                    dates[date_str] = {
                        "status": "available",
                        "available_seats": {
                            "basic_economy": random.randint(0, 20),
                            "economy": random.randint(0, 20),
                            "business": random.randint(0, 15)
                        },
                        "prices": {
                            "basic_economy": int(base_price * CABIN_PRICE_MULTIPLIER["basic_economy"]),
                            "economy": int(base_price * CABIN_PRICE_MULTIPLIER["economy"]),
                            "business": int(base_price * CABIN_PRICE_MULTIPLIER["business"])
                        }
                    }
                elif status == "landed":
                    actual_dep = datetime(current.year, current.month, current.day,
                                         dep_hour, dep_minute) + timedelta(minutes=random.randint(-15, 15))
                    actual_arr = datetime(current.year, current.month, current.day,
                                         arr_hour, arr_minute) + timedelta(minutes=random.randint(-15, 15))
                    dates[date_str] = {
                        "status": "landed",
                        "actual_departure_time_est": actual_dep.strftime("%Y-%m-%dT%H:%M:%S"),
                        "actual_arrival_time_est": actual_arr.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                elif status == "on time":
                    est_dep = datetime(current.year, current.month, current.day,
                                       dep_hour, dep_minute)
                    est_arr = datetime(current.year, current.month, current.day,
                                       arr_hour, arr_minute)
                    dates[date_str] = {
                        "status": "on time",
                        "estimated_departure_time_est": est_dep.strftime("%Y-%m-%dT%H:%M:%S"),
                        "estimated_arrival_time_est": est_arr.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                elif status == "delayed":
                    delay_min = random.randint(15, 120)
                    est_dep = datetime(current.year, current.month, current.day,
                                       dep_hour, dep_minute) + timedelta(minutes=delay_min)
                    est_arr = datetime(current.year, current.month, current.day,
                                       arr_hour, arr_minute) + timedelta(minutes=delay_min)
                    dates[date_str] = {
                        "status": "delayed",
                        "estimated_departure_time_est": est_dep.strftime("%Y-%m-%dT%H:%M:%S"),
                        "estimated_arrival_time_est": est_arr.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                else:  # flying
                    actual_dep = datetime(current.year, current.month, current.day,
                                         dep_hour, dep_minute) + timedelta(minutes=random.randint(-15, 15))
                    est_arr = datetime(current.year, current.month, current.day,
                                       arr_hour, arr_minute) + timedelta(minutes=random.randint(-10, 30))
                    dates[date_str] = {
                        "status": "flying",
                        "actual_departure_time_est": actual_dep.strftime("%Y-%m-%dT%H:%M:%S"),
                        "estimated_arrival_time_est": est_arr.strftime("%Y-%m-%dT%H:%M:%S")
                    }
            else:
                # Future date - available
                base_price = random.randint(80, 400)
                dates[date_str] = {
                    "status": "available",
                    "available_seats": {
                        "basic_economy": random.randint(5, 30),
                        "economy": random.randint(5, 25),
                        "business": random.randint(3, 15)
                    },
                    "prices": {
                        "basic_economy": int(base_price * CABIN_PRICE_MULTIPLIER["basic_economy"]),
                        "economy": int(base_price * CABIN_PRICE_MULTIPLIER["economy"]),
                        "business": int(base_price * CABIN_PRICE_MULTIPLIER["business"])
                    }
                }

            current += timedelta(days=1)

        flights[flight_number] = {
            "origin": origin,
            "destination": dest,
            "flight_number": flight_number,
            "scheduled_departure_time_est": scheduled_departure,
            "scheduled_arrival_time_est": scheduled_arrival,
            "dates": dates
        }

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{target_count} flights")

    return flights


def generate_users_with_llm(
    target_count: int,
    model: str = DEFAULT_MODEL
) -> dict:
    """Generate users with LLM-generated names"""

    users = {}
    existing_user_ids = set()
    existing_emails = set()

    print(f"Generating {target_count} users...")

    # Generate names in batches using LLM
    batch_size = 100
    all_names = []

    for batch_start in range(0, target_count, batch_size):
        batch_count = min(batch_size, target_count - batch_start)

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
            print(f"  Error generating names: {e}, using fallback")
            fallback_first = [
                "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
                "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
                "Thomas", "Sarah", "Christopher", "Karen", "Daniel", "Lisa", "Matthew", "Nancy",
                "Anthony", "Betty", "Mark", "Margaret", "Steven", "Sandra", "Paul", "Ashley",
                "Andrew", "Dorothy", "Joshua", "Kimberly", "Kenneth", "Emily", "Kevin", "Donna",
                "Chen", "Wei", "Yuki", "Raj", "Priya", "Carlos", "Maria", "Ahmed", "Fatima",
                "Hassan", "Mei", "Hiroshi", "Sofia", "Ivan", "Olga", "Lars", "Ingrid",
                "Diego", "Ana", "Miguel", "Elena", "Kenji", "Aiko", "Sanjay", "Ananya",
                "Omar", "Layla", "Tariq", "Nadia", "Viktor", "Natasha", "Kofi", "Amara",
                "Liam", "Noah", "Ethan", "Lucas", "Mason", "Logan", "Ava", "Mia", "Luna",
                "Aria", "Zoe", "Chloe", "Riley", "Nora", "Lily", "Eleanor", "Hannah", "Ella"
            ]
            fallback_last = [
                "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
                "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
                "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
                "Li", "Wang", "Kim", "Patel", "Singh", "Nguyen", "Chen", "Ali", "Khan", "Ahmed",
                "Tanaka", "Yamamoto", "Mueller", "Schmidt", "Fischer", "Weber", "Johansson",
                "Larsson", "Petrov", "Ivanov", "Santos", "Silva", "Costa", "Okafor", "Mensah",
                "Park", "Choi", "Nakamura", "Sato", "Gupta", "Sharma", "Kumar", "Das",
                "Hassan", "Ibrahim", "Mohamed", "Abbas", "Reeves", "Cooper", "Reed", "Bell"
            ]
            for _ in range(batch_count):
                all_names.append({
                    "first_name": random.choice(fallback_first),
                    "last_name": random.choice(fallback_last)
                })

    # Create user records
    for name_data in all_names[:target_count]:
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

        # Membership level
        membership = random.choices(MEMBERSHIP_LEVELS, weights=MEMBERSHIP_WEIGHTS)[0]

        # Saved passengers (0-3)
        num_saved = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
        saved_passengers = []
        for _ in range(num_saved):
            saved_passengers.append({
                "first_name": random.choice(all_names)["first_name"],
                "last_name": random.choice(all_names)["last_name"],
                "dob": generate_dob()
            })

        users[user_id] = {
            "user_id": user_id,
            "name": {
                "first_name": first_name,
                "last_name": last_name
            },
            "address": generate_address(),
            "email": email,
            "dob": generate_dob(),
            "payment_methods": generate_payment_methods(),
            "saved_passengers": saved_passengers,
            "membership": membership,
            "reservations": []
        }

    print(f"  Total users: {len(users)}")
    return users


def find_connecting_flights(flights: dict, origin: str, destination: str, date: str) -> list:
    """Find one-stop connections between origin and destination on a given date"""
    connections = []
    for mid_airport in AIRPORTS:
        if mid_airport == origin or mid_airport == destination:
            continue
        # Find first leg
        first_legs = []
        for f in flights.values():
            if f["origin"] == origin and f["destination"] == mid_airport:
                if date in f["dates"] and f["dates"][date].get("status") == "available":
                    first_legs.append(f)
        # Find second leg
        for first in first_legs:
            for f in flights.values():
                if f["origin"] == mid_airport and f["destination"] == destination:
                    if date in f["dates"] and f["dates"][date].get("status") == "available":
                        connections.append((first, f))
    return connections


def generate_reservations(
    flights: dict,
    users: dict,
    target_count: int
) -> dict:
    """Generate reservations with proper policy compliance"""

    reservations = {}
    existing_ids = set()
    user_list = list(users.keys())

    # Build index of available flights by date and route
    available_flights_by_date = {}
    for flight in flights.values():
        for date, info in flight["dates"].items():
            if info.get("status") == "available":
                key = (flight["origin"], flight["destination"], date)
                if key not in available_flights_by_date:
                    available_flights_by_date[key] = []
                available_flights_by_date[key].append(flight)

    # Also index future/past flights for reservations
    all_flight_dates = []
    for flight in flights.values():
        for date in flight["dates"].keys():
            all_flight_dates.append((flight, date))

    print(f"Generating {target_count} reservations...")

    for i in range(target_count):
        res_id = generate_reservation_id(existing_ids)
        existing_ids.add(res_id)

        # Select user
        user_id = random.choice(user_list)
        user = users[user_id]

        # Trip type
        trip_type = random.choices(
            list(TRIP_TYPE_WEIGHTS.keys()),
            weights=list(TRIP_TYPE_WEIGHTS.values())
        )[0]

        # Cabin class
        cabin = random.choices(
            list(CABIN_WEIGHTS.keys()),
            weights=list(CABIN_WEIGHTS.values())
        )[0]

        # Select origin and destination
        origin = random.choice(AIRPORTS)
        destination = random.choice([a for a in AIRPORTS if a != origin])

        # Select date (can be past or future relative to CURRENT_DATE)
        date = DATE_RANGE_START + timedelta(days=random.randint(0, 29))
        date_str = date.strftime("%Y-%m-%d")

        # Find flights for this route and date
        flight_list = []

        # Try to find a direct flight
        direct_key = (origin, destination, date_str)
        if direct_key in available_flights_by_date:
            selected_flight = random.choice(available_flights_by_date[direct_key])
            price = selected_flight["dates"][date_str]["prices"].get(cabin, 200)
            flight_list.append({
                "origin": origin,
                "destination": destination,
                "flight_number": selected_flight["flight_number"],
                "date": date_str,
                "price": price
            })
        else:
            # Create a flight entry even if not in available index (reservation might have been made earlier)
            # Pick any flight with matching route
            matching = [f for f in flights.values() if f["origin"] == origin and f["destination"] == destination]
            if matching:
                selected_flight = random.choice(matching)
                base_price = random.randint(80, 400)
                price = int(base_price * CABIN_PRICE_MULTIPLIER[cabin])
                flight_list.append({
                    "origin": origin,
                    "destination": destination,
                    "flight_number": selected_flight["flight_number"],
                    "date": date_str,
                    "price": price
                })
            else:
                # Fallback: use any flight and adjust origin/dest
                any_flight = random.choice(list(flights.values()))
                base_price = random.randint(80, 400)
                price = int(base_price * CABIN_PRICE_MULTIPLIER[cabin])
                flight_list.append({
                    "origin": any_flight["origin"],
                    "destination": any_flight["destination"],
                    "flight_number": any_flight["flight_number"],
                    "date": date_str,
                    "price": price
                })
                origin = any_flight["origin"]
                destination = any_flight["destination"]

        # Add return flight for round trips
        if trip_type == "round_trip":
            return_date = date + timedelta(days=random.randint(1, 10))
            return_date_str = return_date.strftime("%Y-%m-%d")
            if return_date <= DATE_RANGE_END:
                return_key = (destination, origin, return_date_str)
                if return_key in available_flights_by_date:
                    return_flight = random.choice(available_flights_by_date[return_key])
                    return_price = return_flight["dates"][return_date_str]["prices"].get(cabin, 200)
                else:
                    # Pick any matching route flight
                    matching_return = [f for f in flights.values()
                                       if f["origin"] == destination and f["destination"] == origin]
                    if matching_return:
                        return_flight = random.choice(matching_return)
                    else:
                        return_flight = random.choice(list(flights.values()))
                    return_price = int(random.randint(80, 400) * CABIN_PRICE_MULTIPLIER[cabin])

                flight_list.append({
                    "origin": destination,
                    "destination": origin,
                    "flight_number": return_flight["flight_number"],
                    "date": return_date_str,
                    "price": return_price
                })

        # Passengers (1-5)
        num_passengers = random.choices([1, 2, 3, 4, 5], weights=[0.35, 0.30, 0.20, 0.10, 0.05])[0]
        passengers = []
        # First passenger is often the user themselves
        passengers.append({
            "first_name": user["name"]["first_name"],
            "last_name": user["name"]["last_name"],
            "dob": user.get("dob", generate_dob())
        })
        # Additional passengers from saved or random
        for j in range(1, num_passengers):
            if user.get("saved_passengers") and j - 1 < len(user["saved_passengers"]):
                passengers.append(user["saved_passengers"][j - 1])
            else:
                passengers.append({
                    "first_name": random.choice(user_list).split("_")[0].capitalize(),
                    "last_name": random.choice(user_list).split("_")[1].capitalize(),
                    "dob": generate_dob()
                })

        # Payment
        user_payment_methods = list(user["payment_methods"].keys())
        if not user_payment_methods:
            # Shouldn't happen, but fallback
            user_payment_methods = ["credit_card_0000000"]

        # Select 1-2 payment methods
        num_payments = min(random.randint(1, 2), len(user_payment_methods))
        selected_payments = random.sample(user_payment_methods, num_payments)

        total_price = sum(f["price"] for f in flight_list) * num_passengers
        payment_history = []
        remaining = total_price
        for idx, pm_id in enumerate(selected_payments):
            if idx == len(selected_payments) - 1:
                amount = remaining
            else:
                amount = random.randint(1, remaining - 1)
                remaining -= amount
            payment_history.append({
                "payment_id": pm_id,
                "amount": amount
            })

        # Baggage
        membership = user.get("membership", "regular")
        free_bags = FREE_BAGS[membership][cabin] * num_passengers
        total_bags = free_bags + random.choices([0, 1, 2, 3], weights=[0.5, 0.3, 0.15, 0.05])[0]
        nonfree_bags = max(0, total_bags - free_bags)

        # Insurance
        insurance = "yes" if random.random() < INSURANCE_RATE else "no"

        # Created time (before the flight date)
        days_before = random.randint(1, 30)
        created_at = date - timedelta(days=days_before)
        created_hour = random.randint(0, 23)
        created_minute = random.randint(0, 59)
        created_second = random.randint(0, 59)
        created_time = created_at.replace(hour=created_hour, minute=created_minute, second=created_second)

        reservation = {
            "reservation_id": res_id,
            "user_id": user_id,
            "origin": origin,
            "destination": destination,
            "flight_type": trip_type,
            "cabin": cabin,
            "flights": flight_list,
            "passengers": passengers,
            "payment_history": payment_history,
            "created_at": created_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_baggages": total_bags,
            "nonfree_baggages": nonfree_bags,
            "insurance": insurance
        }

        reservations[res_id] = reservation

        # Add reservation to user
        users[user_id]["reservations"].append(res_id)

        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{target_count} reservations")

    return reservations


def validate_database(db: GeneratedDatabase) -> list[str]:
    """Validate the generated database for policy compliance"""

    errors = []
    print("Validating database...")

    # Check flights
    flight_numbers = set()
    for fn, flight in db.flights.items():
        if fn != flight["flight_number"]:
            errors.append(f"Flight key {fn} != flight_number {flight['flight_number']}")
        if fn in flight_numbers:
            errors.append(f"Duplicate flight number: {fn}")
        flight_numbers.add(fn)

        if flight["origin"] not in AIRPORTS:
            errors.append(f"Flight {fn} has invalid origin: {flight['origin']}")
        if flight["destination"] not in AIRPORTS:
            errors.append(f"Flight {fn} has invalid destination: {flight['destination']}")
        if flight["origin"] == flight["destination"]:
            errors.append(f"Flight {fn} has same origin and destination")

    print(f"  Flights: {len(db.flights)}")

    # Check users
    user_ids = set()
    for uid, user in db.users.items():
        if uid != user["user_id"]:
            errors.append(f"User key {uid} != user_id {user['user_id']}")
        if uid in user_ids:
            errors.append(f"Duplicate user_id: {uid}")
        user_ids.add(uid)

        if user.get("membership") not in MEMBERSHIP_LEVELS:
            errors.append(f"User {uid} has invalid membership: {user.get('membership')}")

        if not user.get("payment_methods"):
            errors.append(f"User {uid} has no payment methods")

    print(f"  Users: {len(db.users)}")

    # Check reservations
    res_ids = set()
    for rid, res in db.reservations.items():
        if rid != res["reservation_id"]:
            errors.append(f"Reservation key {rid} != reservation_id {res['reservation_id']}")
        if rid in res_ids:
            errors.append(f"Duplicate reservation_id: {rid}")
        res_ids.add(rid)

        if res["user_id"] not in user_ids:
            errors.append(f"Reservation {rid} references non-existent user: {res['user_id']}")

        if res["cabin"] not in CABIN_CLASSES:
            errors.append(f"Reservation {rid} has invalid cabin: {res['cabin']}")

        if res["flight_type"] not in ["one_way", "round_trip"]:
            errors.append(f"Reservation {rid} has invalid flight_type: {res['flight_type']}")

        if not res.get("passengers"):
            errors.append(f"Reservation {rid} has no passengers")
        elif len(res["passengers"]) > 5:
            errors.append(f"Reservation {rid} has more than 5 passengers")

        if not res.get("flights"):
            errors.append(f"Reservation {rid} has no flights")

        # Check payment methods exist in user profile
        user = db.users.get(res["user_id"], {})
        user_pms = set(user.get("payment_methods", {}).keys())
        for payment in res.get("payment_history", []):
            if payment["payment_id"] not in user_pms:
                errors.append(f"Reservation {rid} uses payment {payment['payment_id']} not in user profile")

    print(f"  Reservations: {len(db.reservations)}")

    # Check user.reservations matches actual reservations
    for uid, user in db.users.items():
        user_res = set(user.get("reservations", []))
        actual_res = {rid for rid, r in db.reservations.items() if r["user_id"] == uid}
        if user_res != actual_res:
            missing = actual_res - user_res
            extra = user_res - actual_res
            if missing:
                errors.append(f"User {uid} missing reservations: {list(missing)[:5]}")
            if extra:
                errors.append(f"User {uid} has extra reservations: {list(extra)[:5]}")

    if errors:
        print(f"  Found {len(errors)} validation errors")
    else:
        print("  Validation passed!")

    return errors


def fix_user_reservation_lists(users: dict, reservations: dict) -> dict:
    """Ensure user.reservations lists match the reservations table"""
    user_res_map = {uid: [] for uid in users.keys()}
    for res_id, res in reservations.items():
        if res["user_id"] in user_res_map:
            user_res_map[res["user_id"]].append(res_id)

    for user_id in users:
        users[user_id]["reservations"] = user_res_map.get(user_id, [])

    return users


def save_database(db: GeneratedDatabase, output_path: Path):
    """Save the generated database to JSON"""
    output = {
        "flights": db.flights,
        "users": db.users,
        "reservations": db.reservations
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Database saved to {output_path}")


def main():
    """Main generation pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Airline Database Generator")
    parser.add_argument("--flights", type=int, default=TARGET_FLIGHTS,
                        help=f"Number of flights to generate (default: {TARGET_FLIGHTS})")
    parser.add_argument("--users", type=int, default=TARGET_USERS,
                        help=f"Number of users to generate (default: {TARGET_USERS})")
    parser.add_argument("--reservations", type=int, default=TARGET_RESERVATIONS,
                        help=f"Number of reservations to generate (default: {TARGET_RESERVATIONS})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DB_PATH),
                        help=f"Output path (default: {OUTPUT_DB_PATH})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="LLM model for name generation")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM calls, use fallback names only")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("Airline Database Generator")
    print("=" * 60)
    print(f"  Flights:      {args.flights}")
    print(f"  Users:        {args.users}")
    print(f"  Reservations: {args.reservations}")
    print(f"  Output:       {args.output}")
    print(f"  Seed:         {args.seed}")
    print("=" * 60)

    db = GeneratedDatabase()

    # Step 1: Generate flights
    print("\n[1] Generating flights...")
    db.flights = generate_flights(args.flights)
    print(f"  Generated {len(db.flights)} flights")

    # Step 2: Generate users
    print("\n[2] Generating users...")
    if args.no_llm:
        # Use diverse fallback names without LLM
        fallback_first = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Christopher", "Karen", "Daniel", "Lisa", "Matthew", "Nancy",
            "Anthony", "Betty", "Mark", "Margaret", "Steven", "Sandra", "Paul", "Ashley",
            "Andrew", "Dorothy", "Joshua", "Kimberly", "Kenneth", "Emily", "Kevin", "Donna",
            "Chen", "Wei", "Yuki", "Raj", "Priya", "Carlos", "Maria", "Ahmed", "Fatima",
            "Hassan", "Mei", "Hiroshi", "Sofia", "Ivan", "Olga", "Lars", "Ingrid",
            "Diego", "Ana", "Miguel", "Elena", "Kenji", "Aiko", "Sanjay", "Ananya",
            "Omar", "Layla", "Tariq", "Nadia", "Viktor", "Natasha", "Kofi", "Amara",
            "Liam", "Noah", "Ethan", "Lucas", "Mason", "Logan", "Ava", "Mia", "Luna",
            "Aria", "Zoe", "Chloe", "Riley", "Nora", "Lily", "Eleanor", "Hannah", "Ella"
        ]
        fallback_last = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
            "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
            "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
            "Li", "Wang", "Kim", "Patel", "Singh", "Nguyen", "Chen", "Ali", "Khan", "Ahmed",
            "Tanaka", "Yamamoto", "Mueller", "Schmidt", "Fischer", "Weber", "Johansson",
            "Larsson", "Petrov", "Ivanov", "Santos", "Silva", "Costa", "Okafor", "Mensah",
            "Park", "Choi", "Nakamura", "Sato", "Gupta", "Sharma", "Kumar", "Das",
            "Hassan", "Ibrahim", "Mohamed", "Abbas", "Reeves", "Cooper", "Reed", "Bell"
        ]
        all_names = [{"first_name": random.choice(fallback_first),
                      "last_name": random.choice(fallback_last)}
                     for _ in range(args.users)]

        users = {}
        existing_user_ids = set()
        existing_emails = set()
        for name_data in all_names:
            first_name = name_data["first_name"]
            last_name = name_data["last_name"]
            while True:
                suffix = random.randint(1000, 9999)
                user_id = f"{first_name.lower()}_{last_name.lower()}_{suffix}"
                if user_id not in existing_user_ids:
                    break
            existing_user_ids.add(user_id)
            while True:
                email_suffix = random.randint(1000, 9999)
                email = f"{first_name.lower()}.{last_name.lower()}{email_suffix}@example.com"
                if email not in existing_emails:
                    break
            existing_emails.add(email)
            membership = random.choices(MEMBERSHIP_LEVELS, weights=MEMBERSHIP_WEIGHTS)[0]
            num_saved = random.choices([0, 1, 2, 3], weights=[0.3, 0.4, 0.2, 0.1])[0]
            saved_passengers = []
            for _ in range(num_saved):
                saved_passengers.append({
                    "first_name": random.choice(fallback_first),
                    "last_name": random.choice(fallback_last),
                    "dob": generate_dob()
                })
            users[user_id] = {
                "user_id": user_id,
                "name": {"first_name": first_name, "last_name": last_name},
                "address": generate_address(),
                "email": email,
                "dob": generate_dob(),
                "payment_methods": generate_payment_methods(),
                "saved_passengers": saved_passengers,
                "membership": membership,
                "reservations": []
            }
        db.users = users
        print(f"  Generated {len(db.users)} users (no-llm mode)")
    else:
        db.users = generate_users_with_llm(args.users, model=args.model)

    # Step 3: Generate reservations
    print("\n[3] Generating reservations...")
    db.reservations = generate_reservations(db.flights, db.users, args.reservations)
    print(f"  Generated {len(db.reservations)} reservations")

    # Step 4: Fix user-reservation associations
    print("\n[4] Fixing user-reservation associations...")
    db.users = fix_user_reservation_lists(db.users, db.reservations)

    # Step 5: Validate
    print("\n[5] Validating database...")
    errors = validate_database(db)

    if errors:
        print(f"\nValidation errors ({len(errors)}):")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")

    # Step 6: Save
    print("\n[6] Saving database...")
    save_database(db, Path(args.output))

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"  Flights:      {len(db.flights)}")
    print(f"  Users:        {len(db.users)}")
    print(f"  Reservations: {len(db.reservations)}")
    print(f"  Output:       {args.output}")

    # Stats
    available_count = sum(
        1 for f in db.flights.values()
        for d in f["dates"].values()
        if d.get("status") == "available"
    )
    print(f"  Available flight-dates: {available_count}")
    memberships = {}
    for u in db.users.values():
        m = u.get("membership", "none")
        memberships[m] = memberships.get(m, 0) + 1
    print(f"  Memberships: {memberships}")
    cabins = {}
    for r in db.reservations.values():
        c = r.get("cabin", "unknown")
        cabins[c] = cabins.get(c, 0) + 1
    print(f"  Cabin distribution: {cabins}")


if __name__ == "__main__":
    main()
