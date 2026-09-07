"""
Reusable WoE/IV binning functions for the Home Credit scorecard.
Extracted from 04_scorecard_woe.ipynb so both that notebook and
06_model_packaging.ipynb import from a single source of truth.
"""

import numpy as np
import pandas as pd


def calculate_woe_iv(df, feature_col, target_col='TARGET', n_bins=10, min_bad_per_bin=100, smoothing=0.5):
    """
    Bin a numeric feature into quantile groups (plus an explicit 'Missing' bin),
    then calculate WoE and IV per bin. Applies additive (Laplace) smoothing to
    avoid division by zero when a bin has zero 'good' or 'bad' cases.
    """
    woe_df = pd.DataFrame({
        feature_col: df[feature_col],
        target_col: df[target_col]
    })

    woe_df['bin'] = pd.qcut(woe_df[feature_col], q=n_bins, duplicates='drop').astype('object')
    woe_df.loc[woe_df[feature_col].isnull(), 'bin'] = 'Missing'

    total_good = (woe_df[target_col] == 0).sum()
    total_bad = (woe_df[target_col] == 1).sum()

    grouped = woe_df.groupby('bin', observed=True).agg(
        n_total=(target_col, 'count'),
        n_bad=(target_col, 'sum')
    ).reset_index()

    grouped['n_good'] = grouped['n_total'] - grouped['n_bad']
    grouped['pct_good'] = (grouped['n_good'] + smoothing) / (total_good + smoothing * len(grouped))
    grouped['pct_bad'] = (grouped['n_bad'] + smoothing) / (total_bad + smoothing * len(grouped))
    grouped['woe'] = np.log(grouped['pct_good'] / grouped['pct_bad'])
    grouped['iv_component'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe']

    iv_total = grouped['iv_component'].sum()

    unstable_bins = grouped[grouped['n_bad'] < min_bad_per_bin]
    if len(unstable_bins) > 0:
        print(f"AVISO [{feature_col}]: {len(unstable_bins)} bin(s) com menos de {min_bad_per_bin} casos 'bad', WoE pode ser instável:")
        print(unstable_bins[['bin', 'n_bad']].to_string(index=False))
        print()

    return grouped[['bin', 'n_total', 'n_bad', 'pct_good', 'pct_bad', 'woe', 'iv_component']], iv_total


def calculate_woe_iv_categorical(df, feature_col, target_col='TARGET', min_bad_per_bin=100, smoothing=0.5):
    """
    Calculate WoE and IV for a categorical feature, using each category
    as its own bin. Missing values (if any) get an explicit 'Missing' category.
    Applies additive (Laplace) smoothing to avoid division by zero when a
    category has zero 'good' or 'bad' cases.
    """
    woe_df = pd.DataFrame({
        feature_col: df[feature_col],
        target_col: df[target_col]
    })
    woe_df[feature_col] = woe_df[feature_col].astype('object')
    woe_df[feature_col] = woe_df[feature_col].fillna('Missing')

    total_good = (woe_df[target_col] == 0).sum()
    total_bad = (woe_df[target_col] == 1).sum()

    grouped = woe_df.groupby(feature_col, observed=True).agg(
        n_total=(target_col, 'count'),
        n_bad=(target_col, 'sum')
    ).reset_index().rename(columns={feature_col: 'bin'})

    grouped['n_good'] = grouped['n_total'] - grouped['n_bad']
    grouped['pct_good'] = (grouped['n_good'] + smoothing) / (total_good + smoothing * len(grouped))
    grouped['pct_bad'] = (grouped['n_bad'] + smoothing) / (total_bad + smoothing * len(grouped))
    grouped['woe'] = np.log(grouped['pct_good'] / grouped['pct_bad'])
    grouped['iv_component'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe']

    iv_total = grouped['iv_component'].sum()

    unstable_bins = grouped[grouped['n_bad'] < min_bad_per_bin]
    if len(unstable_bins) > 0:
        print(f"AVISO [{feature_col}]: {len(unstable_bins)} categoria(s) com menos de {min_bad_per_bin} casos 'bad', WoE pode ser instável:")
        print(unstable_bins[['bin', 'n_bad']].to_string(index=False))
        print()

    return grouped.sort_values('woe')[['bin', 'n_total', 'n_bad', 'pct_good', 'pct_bad', 'woe', 'iv_component']], iv_total


def collapse_rare_categories(df, feature_col, target_col='TARGET', min_bad=100):
    """
    Relabel categories with fewer than min_bad 'bad' cases as 'Other_grouped'
    (a name unlikely to collide with existing categories). If the resulting
    grouped bucket is still below min_bad, merge it into the nearest-WoE
    'large' category instead of leaving it unstable.
    """
    series = df[feature_col].astype('object').fillna('Missing')
    bad_counts = df.groupby(series)[target_col].sum()
    rare_categories = bad_counts[bad_counts < min_bad].index.tolist()

    new_series = series.replace(rare_categories, 'Other_grouped')

    # Check if the grouped bucket itself is still below threshold
    new_bad_counts = df.groupby(new_series)[target_col].sum()
    if 'Other_grouped' in new_bad_counts.index and new_bad_counts['Other_grouped'] < min_bad:
        temp_df = pd.DataFrame({feature_col: new_series, target_col: df[target_col]})
        temp_result, _ = calculate_woe_iv_categorical(temp_df, feature_col, target_col, min_bad_per_bin=0)
        grouped_woe = temp_result.loc[temp_result['bin'] == 'Other_grouped', 'woe'].values[0]
        large_bins = temp_result[(temp_result['bin'] != 'Other_grouped') & (temp_result['n_bad'] >= min_bad)].copy()
        large_bins['woe_diff'] = (large_bins['woe'] - grouped_woe).abs()
        nearest_large = large_bins.sort_values('woe_diff').iloc[0]['bin']
        new_series = new_series.replace('Other_grouped', nearest_large)
        print(f"'{feature_col}': Other_grouped ainda instável, fundido com '{nearest_large}'")

    return new_series


def merge_missing_into_nearest_bin(df, feature_col, target_col='TARGET', n_bins=10, smoothing=0.5):
    """
    Bin a numeric feature, then if the Missing bin has an unstable count,
    merge it directly into the numeric bin with the closest WoE, instead
    of keeping Missing as its own bin.
    """
    result, iv = calculate_woe_iv(df, feature_col, target_col, n_bins=n_bins, smoothing=smoothing)

    if 'Missing' not in result['bin'].values:
        return result, iv

    missing_woe = result.loc[result['bin'] == 'Missing', 'woe'].values[0]
    numeric_bins = result[result['bin'] != 'Missing'].copy()
    numeric_bins['woe_diff'] = (numeric_bins['woe'] - missing_woe).abs()
    nearest_bin = numeric_bins.sort_values('woe_diff').iloc[0]['bin']

    # Rebuild the bin series, this time merging nulls directly into nearest_bin's interval
    woe_df = pd.DataFrame({feature_col: df[feature_col], target_col: df[target_col]})
    woe_df['bin'] = pd.qcut(woe_df[feature_col], q=n_bins, duplicates='drop').astype('object')
    woe_df.loc[woe_df[feature_col].isnull(), 'bin'] = nearest_bin

    total_good = (woe_df[target_col] == 0).sum()
    total_bad = (woe_df[target_col] == 1).sum()

    grouped = woe_df.groupby('bin', observed=True).agg(
        n_total=(target_col, 'count'), n_bad=(target_col, 'sum')
    ).reset_index()
    grouped['n_good'] = grouped['n_total'] - grouped['n_bad']
    grouped['pct_good'] = (grouped['n_good'] + smoothing) / (total_good + smoothing * len(grouped))
    grouped['pct_bad'] = (grouped['n_bad'] + smoothing) / (total_bad + smoothing * len(grouped))
    grouped['woe'] = np.log(grouped['pct_good'] / grouped['pct_bad'])
    grouped['iv_component'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe']
    iv_total = grouped['iv_component'].sum()

    print(f"'{feature_col}': Missing (WoE={missing_woe:.4f}) fundido com bin {nearest_bin} (WoE original={numeric_bins.loc[numeric_bins['bin']==nearest_bin,'woe'].values[0]:.4f})")

    return grouped[['bin', 'n_total', 'n_bad', 'pct_good', 'pct_bad', 'woe', 'iv_component']], iv_total


def calculate_woe_iv_with_zero_bin(df, feature_col, target_col='TARGET', n_bins=10, smoothing=0.5, min_bad_per_bin=100):
    """
    Bin a numeric feature with three explicit categories: Missing, Zero,
    and quantile-based bins over the strictly positive values only.
    Useful when a large share of non-null values are exactly zero,
    which would otherwise distort a plain qcut.
    """
    woe_df = pd.DataFrame({feature_col: df[feature_col], target_col: df[target_col]})

    conditions = [
        woe_df[feature_col].isnull(),
        woe_df[feature_col] == 0
    ]
    # Force plain object dtype, not pandas' new strict StringArray,
    # so Interval objects from qcut can be assigned into this column later.
    woe_df['bin'] = pd.Series(
        np.select(conditions, ['Missing', 'Zero'], default=None),
        index=woe_df.index, dtype='object'
    )

    positive_mask = woe_df['bin'].isnull()
    woe_df.loc[positive_mask, 'bin'] = pd.qcut(
        woe_df.loc[positive_mask, feature_col], q=n_bins, duplicates='drop'
    ).astype('object')

    total_good = (woe_df[target_col] == 0).sum()
    total_bad = (woe_df[target_col] == 1).sum()
    grouped = woe_df.groupby('bin', observed=True).agg(
        n_total=(target_col, 'count'), n_bad=(target_col, 'sum')
    ).reset_index()
    grouped['n_good'] = grouped['n_total'] - grouped['n_bad']
    grouped['pct_good'] = (grouped['n_good'] + smoothing) / (total_good + smoothing * len(grouped))
    grouped['pct_bad'] = (grouped['n_bad'] + smoothing) / (total_bad + smoothing * len(grouped))
    grouped['woe'] = np.log(grouped['pct_good'] / grouped['pct_bad'])
    grouped['iv_component'] = (grouped['pct_good'] - grouped['pct_bad']) * grouped['woe']
    iv_total = grouped['iv_component'].sum()

    unstable = grouped[grouped['n_bad'] < min_bad_per_bin]
    if len(unstable) > 0:
        print(f"AVISO [{feature_col}]: {len(unstable)} bin(s) com menos de {min_bad_per_bin} casos 'bad':")
        print(unstable[['bin', 'n_bad']].to_string(index=False))
        print()

    return grouped[['bin', 'n_total', 'n_bad', 'pct_good', 'pct_bad', 'woe', 'iv_component']], iv_total


def extract_edges(result_table):
    """
    Extract plain float bin edges from a WoE table's Interval bins,
    extending the outer edges to -inf/+inf so any future value, however
    extreme, always falls into some bin instead of becoming null.
    """
    intervals = sorted(
            [b for b in result_table['bin'] if isinstance(b, pd.Interval)],
            key=lambda b: b.left
        )
    edges = [intervals[0].left] + [iv.right for iv in intervals]
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def ordered_bin_woe(result_table):
    """Return the WoE of each Interval bin, sorted by the bin's left edge,
    matching the order pd.cut(..., labels=False) will assign at apply time."""
    rows = [(row['bin'], row['woe']) for _, row in result_table.iterrows()
            if isinstance(row['bin'], pd.Interval)]
    rows.sort(key=lambda r: r[0].left)
    return [woe for _, woe in rows]

def apply_frozen_woe(raw_series, spec):
    """
    Transform a raw column into WoE values using a frozen specification
    built strictly from training data. Never recalculates bins or
    categories from the data being transformed, only looks values up.
    Numeric bins are matched by position (pd.cut with labels=False), not
    by Interval-object identity, which is fragile to floating-point
    precision once edges are extended to -inf/+inf.
    """
    if spec['type'] == 'numeric':
        bin_idx = pd.cut(raw_series, bins=spec['edges'], labels=False)
        woe = bin_idx.map(lambda i: spec['bin_woe_ordered'][int(i)] if pd.notnull(i) else np.nan)
        if spec.get('zero_woe') is not None:
            woe[raw_series == 0] = spec['zero_woe']
        if spec.get('missing_woe') is not None:
            woe[raw_series.isnull()] = spec['missing_woe']
        return woe
    else:
        table = spec['table']
        category_map = spec['category_map']
        s = raw_series.astype('object').fillna('Missing')
        if category_map is not None:
            fallback = 'Other_grouped' if 'Other_grouped' in table['bin'].values else 'Missing'
            s = s.map(lambda v: category_map.get(v, fallback))
        woe_map = dict(zip(table['bin'], table['woe']))
        return s.map(woe_map)