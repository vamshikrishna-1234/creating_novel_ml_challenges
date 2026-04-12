# Dataset creation form — fill-in

## 1. Title (Required)

```
Synthetic Session Behavioral Interaction Signal Logs
```

---

## 2. Description (Required)

# Dataset Description

## Overview

This dataset is fully synthetic. It simulates behavioral interaction signal logs from a fictional item-browsing platform. Each record represents a user's interaction with an item during a browsing session, capturing behavioral signals such as engagement intensity, depth of exploration, event type, and sequential position within the session.

The dataset was created to benchmark **latent selection function inference** — determining which item was selected by a hidden decision process, given within-session behavioral signals, user profile features, and item catalog attributes. The task depends on per-interaction behavioral signals within a single session, not on aggregated user-item interaction histories. Approximately 20% of sessions follow a "passive interest" regime where the selected item is driven by user-item feature compatibility rather than engagement signals, creating a dual-regime inference problem.

The dataset contains approximately 261,000 interaction records across 15,000 browsing sessions, involving 3,000 users and a catalog of 500 items across 20 categories. Each session records 6–14 unique items with varying interaction types and behavioral signals. Exactly one item per session was selected. Interaction signals, user profiles, and item features have been obfuscated through normalization, binning, anonymization, and noise injection.

## File Structure

- `interactions.csv` — All session-level behavioral interactions (~261,000 rows)
- `items.csv` — Item catalog with anonymized features (500 rows)
- `users.csv` — User profiles with anonymized features (3,000 rows)
- `purchases.csv` — Ground truth: which item was selected per session (15,000 rows)

## Features

### interactions.csv

| Column | Type | Description |
|--------|------|-------------|
| session_id | int | Unique browsing session identifier |
| user_id | int | User who conducted this session |
| item_id | int | Item interacted with in this session |
| action_type | string | Type of interaction event (view, click, cart, remove) |
| dwell_seconds | float | Time spent on the item during this interaction (seconds) |
| scroll_pct | float | Scroll depth achieved during this interaction (0.0 to 1.0) |
| position | int | Sequential position of this interaction within the session |

### items.csv

| Column | Type | Description |
|--------|------|-------------|
| item_id | int | Unique item identifier |
| category | int | Item category (0–19) |
| price_tier | int | Price bracket (1–5) |
| attr_1 | float | Numeric item attribute |
| attr_2 | float | Numeric item attribute |
| attr_3 | float | Numeric item attribute |

### users.csv

| Column | Type | Description |
|--------|------|-------------|
| user_id | int | Unique user identifier |
| pref_cat_1 | int | Categorical user preference signal (maps to item categories) |
| pref_cat_2 | int | Categorical user preference signal (maps to item categories) |
| pref_cat_3 | int | Categorical user preference signal (maps to item categories) |
| tier_low | int | Lower bound of user's preferred item tier range (1–5) |
| tier_high | int | Upper bound of user's preferred item tier range (1–5) |
| pref_1 | float | Continuous user preference signal |
| pref_2 | float | Continuous user preference signal |
| pref_3 | float | Continuous user preference signal |

### purchases.csv

| Column | Type | Description |
|--------|------|-------------|
| session_id | int | Session identifier |
| purchased_item_id | int | The item_id that was selected in this session |

## Notes

- Each session has exactly one selected item.
- A user may appear in multiple sessions.
- Items may appear across many sessions.
- All data is synthetically generated with a fixed random seed (42) for reproducibility.
- No real individuals, products, or transactions are represented.

---

## 3. Data Files (Required)

Upload all four CSV files: `interactions.csv`, `items.csv`, `users.csv`, `purchases.csv`.

---

## 4. License

**Synthetic data — no external license needed.**

Select: `Other`
