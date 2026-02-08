import os, sys

def creatorofdirs():
    sys.tracebacklimit=0
    media_root = os.path.abspath(os.path.expandvars(os.path.expanduser(os.path.join(os.curdir, "ProbemakerOut"))))
    media_fasta = os.path.join(media_root, "FASTA")
    media_opool = os.path.join(media_root, "OPOOL")
    media_oligo = os.path.join(media_root, "OLIGO")
    media_txt = os.path.join(media_root, "REPORTS")
    blast_root = "blastn"

    os.makedirs(media_root, exist_ok=True)
    os.makedirs(media_fasta, exist_ok=True)
    os.makedirs(media_opool, exist_ok=True)
    os.makedirs(media_oligo, exist_ok=True)
    os.makedirs(media_txt, exist_ok=True)

    return([str(media_root), str(blast_root), str(media_fasta), str(media_opool), str(media_oligo), str(media_txt)])
