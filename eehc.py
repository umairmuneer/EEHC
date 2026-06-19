import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


# ── preprocessing ──────────────────────────────────────────────────────────

def load_chicago(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude', 'Date'])
    df = df[(df['Latitude'] > 41.6) & (df['Latitude'] < 42.1)]
    df = df[(df['Longitude'] > -87.90) & (df['Longitude'] < -87.50)]
    df['Latitude'].fillna(df['Latitude'].median(), inplace=True)
    if 'District' in df.columns:
        df['District'] = df['District'].fillna(method='ffill', limit=1)
    cutoff = df['Date'].dt.year.max() - 1
    return df[df['Date'].dt.year >= cutoff].reset_index(drop=True)


def load_newyork(path):
    df = pd.read_csv(path, low_memory=False)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    if lat_col and lon_col:
        df = df.rename(columns={lat_col: 'Latitude', lon_col: 'Longitude'})
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df[(df['Latitude'] > 40.4) & (df['Latitude'] < 40.95)]
    df = df[(df['Longitude'] > -74.3) & (df['Longitude'] < -73.6)]
    return df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)


def load_lahore(path):
    df = pd.read_csv(path)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    if lat_col and lon_col:
        df = df.rename(columns={lat_col: 'Latitude', lon_col: 'Longitude'})
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df[(df['Latitude'] > 31.2) & (df['Latitude'] < 31.7)]
    df = df[(df['Longitude'] > 74.0) & (df['Longitude'] < 74.6)]
    return df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)


def get_coords(df):
    return df[['Latitude', 'Longitude']].values.astype(np.float64)


# ── ensemble generation ─────────────────────────────────────────────────────

METRICS = ['euclidean', 'manhattan', 'mahalanobis', 'minkowski']
MCS_VALS = [5, 10, 20, 40]
MS_VALS  = [5, 10, 20]


def run_hdbscan(coords, mcs, ms, metric):
    scaler = StandardScaler()
    X = scaler.fit_transform(coords)

    kwargs = dict(min_cluster_size=mcs, min_samples=ms, metric=metric,
                  cluster_selection_method='eom', prediction_data=False)

    if metric == 'mahalanobis':
        cov = np.cov(X.T)
        reg = cov + 0.01 * np.eye(cov.shape[0])
        kwargs['metric_params'] = {'VI': np.linalg.inv(reg)}
    elif metric == 'minkowski':
        kwargs['metric_params'] = {'p': 3}

    try:
        cl = hdbscan.HDBSCAN(**kwargs)
        labels = cl.fit_predict(X)
    except Exception:
        return None, np.nan

    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2 or mask.sum() < 10:
        return labels, np.nan

    try:
        sil = silhouette_score(X[mask], labels[mask])
    except Exception:
        sil = np.nan

    return labels, sil


def build_ensemble(coords, mcs_vals=None, ms_vals=None, metrics=None, verbose=True):
    mcs_vals = mcs_vals or MCS_VALS
    ms_vals  = ms_vals  or MS_VALS
    metrics  = metrics  or METRICS

    partitions, scores = [], []
    combos = list(product(metrics, mcs_vals, ms_vals))

    for i, (m, mcs, ms) in enumerate(combos):
        labels, sil = run_hdbscan(coords, mcs, ms, m)
        if labels is not None:
            partitions.append(labels)
            scores.append(sil)
            if verbose:
                print(f"  [{i+1:3d}/{len(combos)}] metric={m:<12} mcs={mcs:2d} ms={ms:2d}  sil={sil:.3f if not np.isnan(sil) else 'nan'}")

    return partitions, scores


# ── evaluation / filtering ──────────────────────────────────────────────────

def filter_partitions(coords, partitions, scores, tau=0.3, kappa=3, verbose=True):
    scaler = StandardScaler()
    X = scaler.fit_transform(coords)

    selected, sel_scores, sel_idx = [], [], []

    for i, (labels, gsil) in enumerate(zip(partitions, scores)):
        if np.isnan(gsil):
            continue
        mask = labels != -1
        n_cl = len(set(labels[mask]))
        if n_cl < 2 or mask.sum() < 10:
            continue

        try:
            sv = silhouette_samples(X[mask], labels[mask])
        except Exception:
            continue

        n1 = int(np.sum(sv <= 0))
        n2 = int(np.sum((sv > 0) & (sv < tau)))

        if n1 == 0 and n2 <= kappa:
            selected.append(labels)
            sel_scores.append(gsil)
            sel_idx.append(i)
        elif verbose:
            print(f"  rejected partition {i:3d}  n1={n1}  n2={n2}  sil={gsil:.3f}")

    if verbose:
        print(f"\n  retained {len(selected)}/{len(partitions)} partitions")

    return selected, sel_scores, sel_idx


# ── hypergraph construction ─────────────────────────────────────────────────

def build_hypergraph(partitions, n_pts, conn_pct=60.0):
    hyperedges = {}
    weights = {}
    eid = 0

    for labels in partitions:
        for cid in set(labels):
            if cid == -1:
                continue
            members = np.where(labels == cid)[0].tolist()
            if len(members) < 2:
                continue
            key = tuple(sorted(members))
            if key in weights:
                weights[key] += 1
            else:
                weights[key] = 1
                hyperedges[eid] = members
                eid += 1

    if not hyperedges:
        return hyperedges, None

    H = np.zeros((n_pts, len(hyperedges)), dtype=np.float32)
    for idx, (eid, members) in enumerate(hyperedges.items()):
        H[members, idx] = 1.0

    # connectivity-based sparsification
    col_sums = H.sum(axis=0)
    threshold = np.percentile(col_sums, 100 - conn_pct)
    keep = col_sums >= threshold
    H = H[:, keep]

    return hyperedges, H


def partition_hypergraph(H, k):
    if H is None or H.shape[1] < 2:
        return np.zeros(H.shape[0] if H is not None else 1, dtype=int)

    # build affinity from hypergraph
    W = H @ H.T
    diag = np.diag(W)
    norm = np.sqrt(np.outer(diag, diag)) + 1e-10
    A = W / norm

    try:
        sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                                random_state=42, n_init=10)
        return sc.fit_predict(A)
    except Exception:
        return np.zeros(A.shape[0], dtype=int)


def select_best_partition(coords, partitions, H):
    n_pts = len(coords)
    scaler = StandardScaler()
    X = scaler.fit_transform(coords)

    best_labels, best_sil = None, -1.0

    for labels in partitions:
        mask = labels != -1
        n_cl = len(set(labels[mask]))
        if n_cl < 2:
            continue
        try:
            s = silhouette_score(X[mask], labels[mask])
            if s > best_sil:
                best_sil = s
                best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        # fallback: use hypergraph spectral partition
        k = max(2, int(np.sqrt(n_pts / 10)))
        best_labels = partition_hypergraph(H, k)

    return best_labels


# ── kde + boundary smoothing ────────────────────────────────────────────────

def haversine_to_meters(coords):
    lat0 = np.radians(np.mean(coords[:, 0]))
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(lat0)
    result = coords.copy()
    result[:, 0] *= m_per_deg_lat
    result[:, 1] *= m_per_deg_lon
    return result


def kde_density(coords_m, bandwidth):
    if coords_m.shape[0] < 5:
        return np.zeros(coords_m.shape[0])
    bw = bandwidth / np.std(coords_m, axis=0).mean()
    try:
        kde = gaussian_kde(coords_m.T, bw_method=bw)
        return kde(coords_m.T)
    except Exception:
        return np.zeros(coords_m.shape[0])


def cluster_stability(coords_m, labels, cid):
    mask = labels == cid
    if mask.sum() < 3:
        return 0.5
    pts = coords_m[mask]
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    max_d = dists.max() + 1e-10
    return float(1.0 - (dists.mean() / max_d))


def boundary_score(coords_m, labels, cid, bandwidth):
    mask = labels == cid
    if mask.sum() < 3:
        return np.zeros(mask.sum())
    pts = coords_m[mask]
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    max_d = dists.max() + 1e-10
    return dists / max_d  # higher = closer to boundary


def hybrid_density(coords, labels, bandwidth, alpha_global=0.6):
    coords_m = haversine_to_meters(coords)
    n = len(coords)

    kde_scores = np.zeros(n)
    bound_scores = np.zeros(n)

    cluster_ids = [c for c in set(labels) if c != -1]

    for cid in cluster_ids:
        mask = labels == cid
        if mask.sum() < 3:
            continue

        stab = cluster_stability(coords_m, labels, cid)
        alpha_j = 1.0 - stab  # adaptive alpha per cluster

        kde_vals = kde_density(coords_m[mask], bandwidth)
        b_vals = boundary_score(coords_m, labels, cid, bandwidth)

        kde_scores[mask] = kde_vals
        bound_scores[mask] = b_vals

    # normalize
    if kde_scores.max() > 0:
        kde_scores /= kde_scores.max()
    if bound_scores.max() > 0:
        bound_scores /= bound_scores.max()

    # HybridC = alpha * kde + (1-alpha) * (1 - boundary)
    hybrid = alpha_global * kde_scores + (1.0 - alpha_global) * (1.0 - bound_scores)

    noise_mask = labels == -1
    hybrid[noise_mask] = 0.0

    return hybrid


# ── consensus + hotspot detection ──────────────────────────────────────────

def eehc(coords, partitions, bandwidth=700, alpha=0.6, conn_pct=60.0,
         tau_hotspot=0.5, verbose=True):

    n = len(coords)

    if verbose:
        print(f"  building hypergraph from {len(partitions)} partitions...")

    hyperedges, H = build_hypergraph(partitions, n, conn_pct)

    if verbose:
        print(f"  selecting best consensus partition...")

    consensus = select_best_partition(coords, partitions, H)

    if verbose:
        print(f"  computing hybrid density scores...")

    hscores = hybrid_density(coords, consensus, bandwidth, alpha)

    hotspots = (hscores >= tau_hotspot) & (consensus != -1)

    return consensus, hotspots, hscores


# ── validation metrics ──────────────────────────────────────────────────────

def rmsstd(coords, labels):
    scaler = StandardScaler()
    X = scaler.fit_transform(coords)
    total = 0.0
    count = 0
    for cid in set(labels):
        if cid == -1:
            continue
        pts = X[labels == cid]
        if len(pts) < 2:
            continue
        total += np.sum((pts - pts.mean(axis=0)) ** 2)
        count += len(pts)
    return float(np.sqrt(total / count)) if count > 0 else np.nan


def rs_score(coords, labels):
    scaler = StandardScaler()
    X = scaler.fit_transform(coords)
    mask = labels != -1
    if mask.sum() < 2:
        return np.nan
    grand_mean = X[mask].mean(axis=0)
    sst = np.sum((X[mask] - grand_mean) ** 2)
    sse = 0.0
    for cid in set(labels[mask]):
        pts = X[labels == cid]
        sse += np.sum((pts - pts.mean(axis=0)) ** 2)
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def evaluate(coords, labels):
    scaler = StandardScaler()
    X = scaler.fit_transform(coords)
    mask = labels != -1
    n_cl = len(set(labels[mask]))
    n_noise = int((labels == -1).sum())

    out = {
        'n_clusters': n_cl,
        'n_noise': n_noise,
        'noise_pct': 100.0 * n_noise / len(labels)
    }

    if n_cl >= 2 and mask.sum() >= 10:
        try:
            out['silhouette'] = float(silhouette_score(X[mask], labels[mask]))
        except Exception:
            out['silhouette'] = np.nan
        try:
            out['davies_bouldin'] = float(davies_bouldin_score(X[mask], labels[mask]))
        except Exception:
            out['davies_bouldin'] = np.nan
        try:
            out['calinski_harabasz'] = float(calinski_harabasz_score(X[mask], labels[mask]))
        except Exception:
            out['calinski_harabasz'] = np.nan
        out['rmsstd'] = rmsstd(coords, labels)
        out['rs'] = rs_score(coords, labels)

    return out


# ── visualization ───────────────────────────────────────────────────────────

def plot_results(coords, labels, hotspots, title, out_path=None):
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    unique = sorted(c for c in set(labels) if c != -1)
    colors = cm.tab20(np.linspace(0, 1, max(len(unique), 1)))

    ax = axes[0]
    ax.set_title('Cluster Assignments')
    noise = labels == -1
    ax.scatter(coords[noise, 1], coords[noise, 0], c='#cccccc', s=1, alpha=0.3, label='Noise')
    for i, cid in enumerate(unique):
        m = labels == cid
        ax.scatter(coords[m, 1], coords[m, 0], c=[colors[i % len(colors)]], s=3, alpha=0.6)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')

    ax = axes[1]
    ax.set_title('Hotspot Regions')
    non_hot = (~hotspots) & (~noise)
    ax.scatter(coords[non_hot, 1], coords[non_hot, 0], c='#aed6f1', s=2, alpha=0.3, label='Non-hotspot')
    ax.scatter(coords[hotspots, 1], coords[hotspots, 0], c='#e74c3c', s=5, alpha=0.8, label='Hotspot')
    ax.scatter(coords[noise, 1], coords[noise, 0], c='#cccccc', s=1, alpha=0.2, label='Noise')
    ax.set_xlabel('Longitude'); ax.legend(markerscale=3, fontsize=8)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


# ── main pipeline ───────────────────────────────────────────────────────────

CONFIGS = {
    'chicago': {'loader': load_chicago, 'bandwidth': 700},
    'newyork': {'loader': load_newyork, 'bandwidth': 600},
    'lahore':  {'loader': load_lahore,  'bandwidth': 500},
}


def run(dataset, data_path, alpha=0.6, tau=0.3, kappa=3,
        conn_pct=60.0, out_dir='results', verbose=True):

    os.makedirs(out_dir, exist_ok=True)
    cfg = CONFIGS[dataset]

    print(f"\n[1/5] Loading {dataset}...")
    df = cfg['loader'](data_path)
    coords = get_coords(df)
    print(f"      {len(coords):,} records loaded")

    if len(coords) > 50000:
        idx = np.random.choice(len(coords), 50000, replace=False)
        coords = coords[idx]
        print(f"      sampled 50,000 points")

    print(f"\n[2/5] Generating ensemble ({len(METRICS)}x{len(MCS_VALS)}x{len(MS_VALS)} = {len(METRICS)*len(MCS_VALS)*len(MS_VALS)} configs)...")
    partitions, scores = build_ensemble(coords, verbose=verbose)
    print(f"      {len(partitions)} valid partitions")

    print(f"\n[3/5] Filtering partitions (tau={tau}, kappa={kappa})...")
    selected, sel_scores, _ = filter_partitions(coords, partitions, scores,
                                                 tau=tau, kappa=kappa, verbose=verbose)

    if not selected:
        print("      WARNING: no partitions passed filter — relaxing constraints")
        selected, sel_scores, _ = filter_partitions(coords, partitions, scores,
                                                     tau=0.2, kappa=5, verbose=False)

    print(f"\n[4/5] EEHC consensus (bandwidth={cfg['bandwidth']}m, alpha={alpha}, rho={conn_pct}%)...")
    labels, hotspots, hscores = eehc(coords, selected,
                                      bandwidth=cfg['bandwidth'],
                                      alpha=alpha,
                                      conn_pct=conn_pct,
                                      verbose=verbose)

    print(f"\n[5/5] Evaluation metrics:")
    m = evaluate(coords, labels)
    print(f"      Silhouette       : {m.get('silhouette', float('nan')):.4f}")
    print(f"      Davies-Bouldin   : {m.get('davies_bouldin', float('nan')):.4f}")
    print(f"      Calinski-Harabasz: {m.get('calinski_harabasz', float('nan')):.2f}")
    print(f"      RMSSTD           : {m.get('rmsstd', float('nan')):.4f}")
    print(f"      RS               : {m.get('rs', float('nan')):.4f}")
    print(f"      Clusters         : {m.get('n_clusters', 0)}")
    print(f"      Noise            : {m.get('n_noise', 0)} ({m.get('noise_pct', 0):.1f}%)")

    plot_path = os.path.join(out_dir, f'{dataset}_hotspots.png')
    plot_results(coords, labels, hotspots,
                 title=f'Crime Hotspot Detection — {dataset.title()}',
                 out_path=plot_path)
    print(f"\n      Plot saved: {plot_path}")

    out_df = pd.DataFrame({
        'latitude': coords[:, 0],
        'longitude': coords[:, 1],
        'cluster': labels,
        'hotspot': hotspots.astype(int),
        'density_score': hscores
    })
    csv_path = os.path.join(out_dir, f'{dataset}_results.csv')
    out_df.to_csv(csv_path, index=False)
    print(f"      Results saved: {csv_path}")

    return labels, hotspots, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['chicago', 'newyork', 'lahore'])
    p.add_argument('--data_path', required=True)
    p.add_argument('--alpha', type=float, default=0.6)
    p.add_argument('--tau', type=float, default=0.3)
    p.add_argument('--kappa', type=int, default=3)
    p.add_argument('--conn_pct', type=float, default=60.0)
    p.add_argument('--out_dir', default='results')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    run(args.dataset, args.data_path,
        alpha=args.alpha, tau=args.tau, kappa=args.kappa,
        conn_pct=args.conn_pct, out_dir=args.out_dir,
        verbose=not args.quiet)


if __name__ == '__main__':
    main()
