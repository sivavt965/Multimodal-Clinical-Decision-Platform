"""
seed_auth_users.py — Create Supabase Auth users for the 4 demo accounts.

Run once after Phase 5 is activated:
    py seed_auth_users.py

Uses the SERVICE_ROLE key to bypass email confirmation. Passwords are
set to the values in DEMO_USERS below — change them before deploying to
a non-demo environment.

Each created auth user's email matches the `email` column in our `users`
table, which is how the backend bridges Supabase auth.uid() → app role.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    from supabase import create_client
except ImportError:
    print("ERROR: `supabase` package not installed. Run: pip install supabase")
    sys.exit(1)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SERVICE_ROLE_KEY:
    print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in backend/.env")
    sys.exit(1)

# Demo users — email must match the `users` table rows seeded by seed_demo_users.py
DEMO_USERS = [
    {"email": "dr.smith@hospital.org",   "password": "Demo1234!", "full_name": "Dr. Alice Smith"},
    {"email": "dr.johnson@hospital.org", "password": "Demo1234!", "full_name": "Dr. Ben Johnson"},
    {"email": "sarah.lee@hospital.org",  "password": "Demo1234!", "full_name": "Sarah Lee"},
    {"email": "ops@hospital.org",        "password": "Demo1234!", "full_name": "System Operator"},
]

client = create_client(SUPABASE_URL, SERVICE_ROLE_KEY)

print("Creating Supabase Auth users for demo accounts...\n")
for user in DEMO_USERS:
    try:
        # admin.create_user skips email confirmation (service role only)
        resp = client.auth.admin.create_user({
            "email": user["email"],
            "password": user["password"],
            "email_confirm": True,
            "user_metadata": {"full_name": user["full_name"]},
        })
        print(f"  [OK] Created: {user['email']}  (uid={resp.user.id})")
    except Exception as exc:
        msg = str(exc)
        if "already been registered" in msg or "already exists" in msg.lower():
            print(f"  [SKIP] Already exists: {user['email']}")
        else:
            print(f"  [ERROR] {user['email']}: {exc}")

print("\nDone. All demo users can now log in at /login with password: Demo1234!")
print("Change passwords in production before going live.")
