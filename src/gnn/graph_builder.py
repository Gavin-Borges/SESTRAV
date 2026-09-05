import os
from typing import Sequence

import torch

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
MAX_PEPTIDE_LEN: int = 11  # longest supported peptide (9-11-mer; pad shorter seqs)


def allele_cache_key(allele: str) -> str:
    """Convert 'HLA-A*02:01' -> 'A0201' for use in filenames."""
    return allele.replace("HLA-", "").replace("*", "").replace(":", "")


def structural_cache_filename(peptide: str, allele: str) -> str:
    """Canonical filename for one peptide+allele structural distance matrix.

    Peptide backbone geometry is groove-dependent, so the cache is keyed by the
    pair, not by the peptide alone. scripts/run_pandora_structures.py imports this
    function so that the writer and the reader cannot drift apart again.
    """
    return f"{peptide}_{allele_cache_key(allele)}_dist.pt"


def report_structural_cache_coverage(
    peptides: Sequence[str],
    alleles: Sequence[str],
    cache_dir: str,
) -> int:
    """Report how many (peptide, allele) pairs the structural cache actually reaches.

    A pair with no cached distance matrix is not dropped or raised on: it falls
    back to GraphBuilder.build_chain_adj, which keeps every row addressable and
    is the right contract. The cost is that a total miss is indistinguishable
    from a total hit at the tensor level, so a run configured with
    use_spatial_adj could train entirely on chain graphs while its artifact
    recorded a structural cache directory. That is the defect this reports: the
    writer emitted '{peptide}_{allele_key}_dist.pt' while the reader looked for
    '{peptide}_dist.pt', and every lookup missed in total silence.

    Counted ONCE over the whole panel rather than per row, deliberately.
    build_spatial_adj is called from GraphPeptideDataset.__getitem__, once per
    row per epoch, so a per-row warning on a 35k-row corpus is ~35,000 stderr
    lines per epoch and warnings.warn cannot dedup them: its key is
    (text, category, lineno), so a message naming the peptide is unique every
    time. This is the shape src/train_classifier.py's binding-coverage report
    already uses for the same problem - accumulate a count against the panel
    total, emit one line - and the output here is bounded by the number of
    datasets built, never by corpus size.

    Uses print rather than the logging module, and reports unconditionally, for
    the same reason that one does: silence was the defect.

    Returns:
        The number of pairs with no cached matrix.
    """
    total = len(peptides)
    if total == 0:
        return 0
    try:
        cached = set(os.listdir(cache_dir))
    except OSError:
        # An unreadable or absent cache directory is a 100% miss, which is
        # exactly what the caller needs told rather than an exception.
        cached = set()
    missing = sum(
        1
        for peptide, allele in zip(peptides, alleles)
        if structural_cache_filename(peptide, allele) not in cached
    )
    covered = total - missing
    prefix = "Structural cache coverage" if missing == 0 else "WARNING: structural cache coverage"
    print(
        f"{prefix}: {covered}/{total} rows ({covered / total:.1%}) found in "
        f"{cache_dir}; {missing} fell back to the chain adjacency"
    )
    return missing


class GraphBuilder:
    """Builds chain graph representations from peptide sequences."""

    @staticmethod
    def build_chain_adj(max_len: int = MAX_PEPTIDE_LEN) -> torch.Tensor:
        """
        Builds a normalized adjacency matrix for a 1D sequence chain graph.
        """
        adj = torch.zeros((max_len, max_len), dtype=torch.float32)
        for i in range(max_len):
            adj[i, i] = 1.0  # Self-loop
            if i > 0:
                adj[i, i - 1] = 1.0
            if i < max_len - 1:
                adj[i, i + 1] = 1.0

        # Degree normalization (D^-0.5 * A * D^-0.5)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return norm_adj

    @staticmethod
    def build_spatial_adj(
        peptide: str,
        allele: str,
        cache_dir: str,
        max_len: int = MAX_PEPTIDE_LEN,
        distance_threshold: float = 8.0,
    ) -> torch.Tensor:
        """
        Builds a normalized adjacency matrix using pre-computed 3D spatial distances
        (e.g. from AlphaFold PDBs).
        Edges are created if distance <= threshold, and weighted as 1/distance.

        Args:
            peptide: The sequence to look up in the structural cache.
            allele: The MHC allele the structure was modelled against, e.g.
                'HLA-A*02:01'. The cache is keyed by the pair; see
                structural_cache_filename.
            cache_dir: Directory holding '{peptide}_{allele_key}_dist.pt' matrices.
            max_len: Padding length.
            distance_threshold: Angstrom cutoff for edge creation.
        """
        cache_path = os.path.join(cache_dir, structural_cache_filename(peptide, allele))

        # Fallback to chain graph if structure is not cached. Silent BY DESIGN at
        # this level: this runs once per row per epoch, so anything emitted here
        # scales with the corpus. The miss count is reported once per panel by
        # report_structural_cache_coverage, which GraphPeptideDataset calls.
        if not os.path.exists(cache_path):
            return GraphBuilder.build_chain_adj(max_len)

        # Load precomputed pairwise distance matrix (L x L)
        dist_matrix = torch.load(cache_path, weights_only=True)  # nosec B614 - own precomputed distance matrix; weights_only=True enforced
        L = dist_matrix.size(0)

        adj = torch.zeros((max_len, max_len), dtype=torch.float32)

        # Create weighted edges
        for i in range(min(L, max_len)):
            for j in range(min(L, max_len)):
                if i == j:
                    adj[i, j] = 1.0  # Self-loop
                else:
                    d = dist_matrix[i, j].item()
                    if d <= distance_threshold and d > 0:
                        adj[i, j] = 1.0 / d

        # Degree normalization
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return norm_adj

    @staticmethod
    def build_pyg_chain_graph(max_len: int = MAX_PEPTIDE_LEN):
        """Build PyG-format edge_index and edge_attr for a 1D chain graph with self-loops.

        Edge features (3-dim one-hot):
            [1, 0, 0] = self-loop  (i → i)
            [0, 1, 0] = forward    (i → i+1)
            [0, 0, 1] = backward   (i → i-1)

        Returns:
            edge_index: LongTensor  (2, num_edges) - num_edges = max_len + 2*(max_len-1)
            edge_attr:  FloatTensor (num_edges, 3)
        """
        src, dst, attrs = [], [], []
        for i in range(max_len):
            src.append(i)
            dst.append(i)
            attrs.append([1.0, 0.0, 0.0])
            if i < max_len - 1:
                src.append(i)
                dst.append(i + 1)
                attrs.append([0.0, 1.0, 0.0])
                src.append(i + 1)
                dst.append(i)
                attrs.append([0.0, 0.0, 1.0])
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
        return edge_index, edge_attr

    @staticmethod
    def sequence_to_node_features(seq: str, max_len: int = MAX_PEPTIDE_LEN) -> torch.Tensor:
        """Convert peptide to node feature matrix (max_len, num_features)."""
        # Features per node: one-hot encoded AA (size 20)
        features = torch.zeros((max_len, 20))
        for i, aa in enumerate(seq[:max_len]):
            if aa in AA_TO_IDX:
                features[i, AA_TO_IDX[aa]] = 1.0
        return features
