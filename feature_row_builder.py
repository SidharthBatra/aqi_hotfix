"""
feature_row_builder.py

Single shared function for building schema-conformant rows to insert into
the aqi_karachi_features v4 feature group. Used identically by all three
places that write into it:
  - fetch_features.py's live current-hour insert
  - fetch_features.py's self-healing gap-fill (fetch_healing_rows)
  - backfill_gap.py's manual one-off gap-fill CLI

WHY THIS EXISTS (2026-09-13 incident): three independent code paths each
built rows for the same feature group, and the combination of two of them
running in the same script invocation -- a self-heal that reached all the
way to the CURRENT hour, immediately followed by that same hour's live
insert -- produced rows with aqi_change_rate/pm25_change_rate silently
NaN. The live path's "find the previous row" lookup found the row the
self-heal had *just* inserted for the SAME hour (hours_elapsed == 0
between "previous" and "current"), not the actual prior hour. Both
change-rate columns went null together on exactly those hours, and
because 'aqi' itself was still present, the write-time gap detection
(determine_gap(), which only checks 'aqi') never flagged them -- they sat
silently broken until the dashboard tried to score them.

Three separately-maintained row builders computing the same derived
columns is the actual defect, not any one of them being wrong in
isolation -- it will keep producing this class of bug as long as it
stands. This module is the fix: one function, called by all three paths,
that takes trailing context plus whichever new hour(s) need rows -- one
(live) or many (a healed gap, or a heal-plus-live batch in a single run)
-- and computes every derived column, including change rates, across the
WHOLE batch in timestamp order. See build_feature_rows()'s docstring for
the specific rule this imposes on callers that need to heal AND write a
live hour in the same run.
"""

import pandas as pd

from compute_true_aqi import (
    POLLUTANT_COLUMNS,
    categorize,
    clean_sentinel_values,
    compute_sub_indices,
)

TIMESTAMP_COL = "timestamp_utc"

# Every column aqi_karachi_features v4 defines (confirmed live against the
# feature group's schema on Hopsworks, 2026-09-13 -- `fg.columns` is the
# source of truth if this ever needs re-checking). The Step 3 write-time
# guard (assert_row_schema_complete) checks every row has ALL of these
# before it's allowed to reach fg.insert().
FEATURE_GROUP_COLUMNS = [
    "timestamp_utc", "hour", "day", "month", "year", "day_of_week",
    "aqi_index_openweather_1to5", "co", "no", "no2", "o3", "so2",
    "pm2_5", "pm10", "nh3",
    "om_pm2_5", "om_pm10", "om_co", "om_no2", "om_so2", "om_o3", "om_us_aqi",
    "pm25_source_diff", "pm25_source_diff_pct",
    "aqi_change_rate", "pm25_change_rate",
    "aqi", "aqi_dominant_pollutant", "aqi_category",
    "row_id",
]

# Columns that genuinely cannot be computed by ANY of the three write
# paths here -- Open-Meteo is a backfill-only cross-check source (see
# backfill_historical.py), never available live or from OpenWeather's
# History API -- so these are always explicitly set to None rather than
# silently absent. train_model.py's EXCLUDE_COLUMNS already excludes them
# from model features for exactly this reason, so their nullness is never
# something the Step 3 guard should reject.
ALWAYS_NULL_COLUMNS = [
    "om_pm2_5", "om_pm10", "om_co", "om_no2", "om_so2", "om_o3", "om_us_aqi",
    "pm25_source_diff", "pm25_source_diff_pct",
]

# Every column the Hopsworks feature group schema expects as 'double' --
# a clean row (e.g. nh3=0, aqi_index=3) otherwise gets inferred as
# int/bigint by pandas, which Hopsworks then rejects as a schema mismatch.
FLOAT_COLUMNS = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "aqi_index_openweather_1to5", "aqi",
    "aqi_change_rate", "pm25_change_rate",
] + ALWAYS_NULL_COLUMNS


def build_feature_rows(context_df, new_readings_df):
    """Computes complete, schema-conformant feature-group rows for one or
    more new hours, given trailing context.

    Args:
        context_df: existing feature-group rows already committed to
            Hopsworks (or already-computed rows from earlier in the SAME
            call -- see the batching note below). Must have TIMESTAMP_COL
            plus at least POLLUTANT_COLUMNS.values(); only used for
            trailing rolling-window/change-rate context, never modified or
            reinserted itself. May be empty (a cold start).
        new_readings_df: one row per new hour to build, with TIMESTAMP_COL,
            POLLUTANT_COLUMNS.values(), and 'aqi_index' (OpenWeather's raw
            1-5 category -- renamed to aqi_index_openweather_1to5 in the
            output). Row order doesn't matter; rows are sorted by
            timestamp before any derived column is computed.

    Returns:
        A DataFrame with every FEATURE_GROUP_COLUMNS column filled in
        (in schema order), ready for fg.insert().

    Change rates are computed via .diff() over the FULL combined (context
    + new) series in timestamp order, so:
      - a multi-hour healed batch gets internally-consistent rates between
        its own rows, not just against the last pre-existing context row;
      - a single live hour gets a real rate against the most recent
        context row, the normal case;
      - a row with genuinely no predecessor anywhere (the very first row
        of the entire dataset -- context_df empty AND this is the
        earliest row in new_readings_df) gets a NaN rate, matching
        compute_true_aqi.py's df["aqi"].diff() on the original backfill
        (whose first row is NaN by that same construction) -- training
        and serving stay consistent on what "no predecessor" means.

    CALLERS THAT NEED BOTH A HEALED BATCH AND A LIVE HOUR IN THE SAME RUN
    MUST PASS THEM TOGETHER in one new_readings_df, not as two separate
    calls where the second call's context_df is built from the first
    call's own output. Two separate calls reintroduces the exact
    "previous row == this row" collision this function exists to
    eliminate (see module docstring's incident writeup). See
    fetch_features.py's main() for the correct pattern.
    """
    context_cols = [TIMESTAMP_COL] + list(POLLUTANT_COLUMNS.values())
    if context_df is not None and not context_df.empty:
        ctx = context_df[context_cols].copy()
    else:
        ctx = pd.DataFrame(columns=context_cols)

    new = new_readings_df.copy()
    new[TIMESTAMP_COL] = pd.to_datetime(new[TIMESTAMP_COL], utc=True)
    new = new.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    combined = pd.concat([ctx, new[context_cols]], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    combined = clean_sentinel_values(combined)
    combined_indexed = combined.set_index(TIMESTAMP_COL)

    sub_indices = compute_sub_indices(combined_indexed)
    aqi = sub_indices.max(axis=1)
    dominant = sub_indices.idxmax(axis=1)
    category = aqi.apply(categorize)

    # Normalized per elapsed hour, not per row -- correct even across a
    # gap the context itself still has a hole in (e.g. cold-start context).
    hours_elapsed = combined_indexed.index.to_series().diff().dt.total_seconds() / 3600.0
    aqi_rate = (aqi.diff() / hours_elapsed).round(4)
    pm25_rate = (combined_indexed["pm2_5"].diff() / hours_elapsed).round(4)

    derived = pd.DataFrame({
        "aqi": aqi,
        "aqi_dominant_pollutant": dominant,
        "aqi_category": category,
        "aqi_change_rate": aqi_rate,
        "pm25_change_rate": pm25_rate,
    })

    out = new.set_index(TIMESTAMP_COL).join(derived, how="left").reset_index()

    if "aqi_index" in out.columns:
        out = out.rename(columns={"aqi_index": "aqi_index_openweather_1to5"})
    for col in ALWAYS_NULL_COLUMNS:
        out[col] = None

    # .dt.hour/.day/etc. default to int32 on Windows (numpy's platform
    # default int type) but int64 on Linux/CI -- the Hopsworks feature
    # group schema expects 'bigint' (int64) for these regardless of which
    # platform built the row, so cast explicitly rather than relying on
    # whatever the host platform's default int width happens to be
    # (confirmed directly: this insert failed schema validation on
    # Windows with "expected type: 'bigint', derived from input: 'int'").
    out["hour"] = out[TIMESTAMP_COL].dt.hour.astype("int64")
    out["day"] = out[TIMESTAMP_COL].dt.day.astype("int64")
    out["month"] = out[TIMESTAMP_COL].dt.month.astype("int64")
    out["year"] = out[TIMESTAMP_COL].dt.year.astype("int64")
    out["day_of_week"] = out[TIMESTAMP_COL].dt.dayofweek.astype("int64")
    # Derived from the (already hour-floored) Unix timestamp, not
    # sequential -- idempotent on re-run: re-building a row for an hour
    # that already exists reproduces the same row_id and upserts in place
    # rather than inserting a near-duplicate.
    out["row_id"] = out[TIMESTAMP_COL].apply(lambda ts: int(ts.timestamp()))

    misaligned = out[TIMESTAMP_COL].dt.minute.ne(0) | out[TIMESTAMP_COL].dt.second.ne(0)
    assert not misaligned.any(), (
        f"{int(misaligned.sum())} row(s) are not hour-aligned -- refusing to build. "
        f"First offender: {out.loc[misaligned, TIMESTAMP_COL].iloc[0]}"
    )

    for col in FLOAT_COLUMNS:
        out[col] = out[col].astype(float)

    return out[FEATURE_GROUP_COLUMNS]


def assert_row_schema_complete(row_df):
    """Step 3 write-time guard: fail loudly, BEFORE fg.insert(), if a row
    is missing a feature-group column entirely, or is null in a column
    the models actually train on -- rather than silently writing a row
    that only turns out to be unscoreable days later on the dashboard.
    This is the same class of guard as build_feature_rows()'s
    hour-alignment assert; it exists so this exact bug (change rates
    silently null) is caught here, at write time, instead of downstream.

    Deliberately does NOT use train_model.get_feature_columns()'s
    df.select_dtypes(include=[np.number]) filtering to decide what's
    "required" -- that filter silently drops a column if it happens to be
    object-dtyped (e.g. a column of all-None before the float cast
    build_feature_rows() normally applies), which would make this exact
    guard blind to the exact bug it exists to catch. Required columns are
    instead computed directly from the feature group's own schema, using
    train_model.EXCLUDE_COLUMNS (the same identifier/categorical columns
    that aren't feature inputs) as the single source of truth for what to
    skip -- dtype-independent, so it can't be fooled by an unexpected
    dtype the way get_feature_columns() can.

    ALWAYS_NULL_COLUMNS are legitimately always null (see their own
    docstring) and are exempt. A row with genuinely no predecessor
    anywhere (the very first row of the whole dataset) legitimately gets
    a null aqi_change_rate/pm25_change_rate too (see build_feature_rows'
    docstring) -- this is not a case either fetch_features.py or
    backfill_gap.py ever actually hits, since both only ever run against
    an already-populated feature group; a genuinely empty feature group is
    bootstrapped once via backfill_historical.py + compute_true_aqi.py +
    ingest_to_hopsworks.py, a separate path this guard does not cover.

    A row whose 'aqi' ITSELF is null is a separate, pre-existing, and
    deliberately tolerated case -- OpenWeather (live or History API)
    didn't have enough pollutant inputs for that hour to compute anything
    at all (see backfill_gap.py's "missing pollutant inputs" note).
    train_model.py's hourly-grid reindexing and dashboard.py's short-gap
    interpolation already handle these documented gaps by design; failing
    loudly on them here would block a long-accepted, intentional code
    path rather than catch a defect, so such rows are exempt from this
    check entirely. The guard instead focuses on rows where 'aqi' WAS
    successfully computed but a column that's always derivable once 'aqi'
    exists (a change rate, a raw pollutant reading) came out null anyway
    -- that combination is never legitimate, and is exactly the
    2026-09-13 bug this guard exists to catch."""
    missing_cols = [c for c in FEATURE_GROUP_COLUMNS if c not in row_df.columns]
    assert not missing_cols, f"Row(s) missing feature-group column(s): {missing_cols}"

    from train_model import EXCLUDE_COLUMNS

    required = [
        c for c in FEATURE_GROUP_COLUMNS
        if c != TIMESTAMP_COL and c not in ALWAYS_NULL_COLUMNS and c not in EXCLUDE_COLUMNS
    ]
    checkable = row_df[row_df["aqi"].notna()]
    if checkable.empty:
        return

    bad = checkable[required].isna().any(axis=1)
    if bad.any():
        bad_rows = checkable.loc[bad, [TIMESTAMP_COL] + required]
        offenders = {
            str(r[TIMESTAMP_COL]): [c for c in required if pd.isna(r[c])]
            for _, r in bad_rows.iterrows()
        }
        raise AssertionError(
            f"Refusing to insert row(s) with null required feature(s): {offenders}"
        )
