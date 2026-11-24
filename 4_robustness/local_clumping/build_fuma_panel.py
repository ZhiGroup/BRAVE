"""Rebuild FUMA's 1000 Genomes Phase 3 EUR reference panel.

Our current panel is the MAGMA `g1000_eur.zip` build: 503 samples but only
8,550,156 variants. FUMA's is the same 503 samples with 25,063,419 variants, and
that 2.9x gap is the sole remaining source of disagreement in the locus
reproduction (see VALIDATION.md).

Follows FUMA's stated recipe verbatim (https://fuma.ctglab.nl/tutorial):

    "Multi allelic SNPs were first split into separate columns"
    "VCF files were then converted to PLINK bfile (PLINK v1.9). Any CNVs were
     removed, while any indels were kept."
    "SNPs with MAF=0 were excluded for each population."

plink2 does the splitting (`--make-pgen multiallelics=-`), so no bcftools is
needed. CNVs carry symbolic ALT alleles (<CN0>, <INS:ME:ALU>, ...) which plink2
skips on import; indels are ordinary REF/ALT strings and are kept. MAF=0 is
enforced with `--mac 1`, which drops variants monomorphic among the 503 EUR
samples -- those carry no LD information.

Output is one bfile per chromosome, which `fuma_clump.py --bfile-pattern`
consumes via its {} placeholder. That is also far cheaper than the single merged
file: plink reads ~1/22 as much per chromosome.

Chromosomes are processed strictly one at a time -- the reference data sits on a
shared NFS mount and parallel downloads or builds are not worth the risk.

Usage:
  python build_fuma_panel.py --chroms 22            # pilot
  python build_fuma_panel.py                        # all of 1-22
  python build_fuma_panel.py --keep-vcf             # do not delete VCFs
"""
from __future__ import print_function

import argparse
import os
import subprocess
import sys
import time

BASE_URL = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502")
VCF_TMPL = ("ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b"
            ".20130502.genotypes.vcf.gz")
PANEL_FILE = "integrated_call_samples_v3.20130502.ALL.panel"
PLINK2 = ("<path/to>/"
          "plink2_linux_x86_64_20260110/plink2")


def run(cmd, label):
    print("[{0}] {1}".format(label, " ".join(cmd)), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        print(out[-4000:], flush=True)
        raise RuntimeError("{0} failed (exit {1})".format(label,
                                                          proc.returncode))
    print("[{0}] ok in {1:.0f}s".format(label, time.time() - t0), flush=True)
    return out


def write_eur_keep(root):
    """plink2 --keep wants one sample ID per line; 1000G bfiles are single-ID."""
    panel = os.path.join(root, PANEL_FILE)
    if not os.path.exists(panel):
        run(["curl", "-s", "-m", "300", "-o", panel,
             "{0}/{1}".format(BASE_URL, PANEL_FILE)], "panel")
    eur = []
    with open(panel) as fh:
        header = fh.readline().split("\t")
        sup = header.index("super_pop")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > sup and f[sup] == "EUR":
                eur.append(f[0])
    keep = os.path.join(root, "eur_samples.txt")
    with open(keep, "w") as fh:
        fh.write("#IID\n")
        fh.write("\n".join(eur) + "\n")
    print("[panel] {0} EUR samples -> {1}".format(len(eur), keep), flush=True)
    return keep, len(eur)


def build_chrom(root, chrom, keep, keep_vcf):
    vcf_name = VCF_TMPL.format(c=chrom)
    vcf = os.path.join(root, "vcf", vcf_name)
    out = os.path.join(root, "EUR.chr{0}".format(chrom))
    tmp = os.path.join(root, "tmp_chr{0}".format(chrom))

    if os.path.exists(out + ".bed"):
        print("[chr{0}] already built, skipping".format(chrom), flush=True)
        return None

    if not os.path.exists(vcf):
        run(["wget", "-q", "-c", "-O", vcf,
             "{0}/{1}".format(BASE_URL, vcf_name)], "chr{0} download".format(chrom))
    print("[chr{0}] vcf {1:.2f} GB".format(
        chrom, os.path.getsize(vcf) / 1e9), flush=True)

    # Split multi-allelic records and subset to EUR. IDs are assigned here, not
    # later: exclusion and clumping both address variants by ID, and 1000G
    # leaves many as '.'. Splitting also duplicates the source rsID across the
    # resulting rows, so --rm-dup is needed for plink 1.9, which aborts outright
    # on a repeated ID.
    run([PLINK2, "--vcf", vcf, "--keep", keep,
         "--make-pgen", "multiallelics=-",
         "--set-missing-var-ids", "@:#:$r:$a",
         "--new-id-max-allele-len", "1000", "truncate",
         "--rm-dup", "force-first",
         "--out", tmp], "chr{0} split".format(chrom))

    # "Any CNVs were removed": 1000G carries these as symbolic ALT alleles
    # (<CN0>, <CN2>, <INS:ME:ALU>, ...). plink2 does NOT drop them on import --
    # it imports them verbatim, and they later collide on duplicate IDs. Build
    # an explicit exclusion list from the pvar. Indels are ordinary REF/ALT
    # strings and are untouched by this, exactly as FUMA specifies.
    excl = tmp + ".symbolic"
    n_sym = 0
    with open(tmp + ".pvar") as fh, open(excl, "w") as out_fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if "<" in f[3] or "<" in f[4]:
                out_fh.write(f[2] + "\n")
                n_sym += 1
    print("[chr{0}] {1:,} symbolic/CNV variants excluded".format(chrom, n_sym),
          flush=True)

    # --mac 1 implements "SNPs with MAF=0 were excluded". IDs are assigned to
    # any remaining '.' entries so plink 1.9 does not later abort on duplicates;
    # --rm-dup catches genuinely repeated rsIDs.
    # --rm-dup again, deliberately: in the split step it runs before
    # --set-missing-var-ids, so it cannot see collisions among the IDs that step
    # generates. Two distinct indels at one position can reduce to the same
    # chr:pos:ref:alt string, and plink 1.9 aborts the whole chromosome on the
    # first duplicate it meets.
    cmd = [PLINK2, "--pfile", tmp]
    if n_sym:
        cmd += ["--exclude", excl]
    cmd += ["--rm-dup", "force-first",
            "--mac", "1", "--make-bed", "--out", out]
    run(cmd, "chr{0} bed".format(chrom))

    for ext in (".pgen", ".pvar", ".psam", ".log", ".symbolic"):
        p = tmp + ext
        if os.path.exists(p):
            os.remove(p)
    if not keep_vcf and os.path.exists(vcf):
        os.remove(vcf)
        print("[chr{0}] removed vcf".format(chrom), flush=True)

    n = sum(1 for _ in open(out + ".bim"))
    print("[chr{0}] {1:,} variants retained".format(chrom, n), flush=True)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="<path/to>/g1000_phase3")
    ap.add_argument("--chroms", nargs="+", type=int,
                    default=list(range(1, 23)))
    ap.add_argument("--keep-vcf", action="store_true")
    args = ap.parse_args()

    for sub in ("", "vcf"):
        d = os.path.join(args.root, sub)
        if not os.path.isdir(d):
            os.makedirs(d)

    keep, n_eur = write_eur_keep(args.root)
    if n_eur != 503:
        print("[WARN] expected 503 EUR samples, got {0}".format(n_eur),
              flush=True)

    total = 0
    for chrom in args.chroms:
        n = build_chrom(args.root, chrom, keep, args.keep_vcf)
        if n:
            total += n

    print("\n{0}\nbuilt {1} chromosome(s), {2:,} variants this run".format(
        "=" * 60, len(args.chroms), total))
    print("FUMA's EUR panel contains 25,063,419 variants genome-wide.")
    print("panel prefix pattern: {0}/EUR.chr{{}}".format(args.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
