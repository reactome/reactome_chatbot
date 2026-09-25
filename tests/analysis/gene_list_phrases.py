"""Phrasings for the gene-list recogniser, sized, and where each came from.

- REVIEW_*: written by an adversarial review of the first version, which
  fired on 14 of the 44 questions and missed or trimmed 20 of the 25
  requests. The recogniser was then rewritten against these.
- HELD_OUT_*: written after that rewrite and never tuned against. They are
  the honest measure: 15/15 and 0/20 when added.

A positive is exact: the identifiers read, in order. A request that reads
the wrong list is a failure even if it fires.
"""

REVIEW_REQUESTS: list[tuple[str, list[str]]] = [
    ("run ORA on EGFR KRAS BRAF PTEN NRAS", ["EGFR", "KRAS", "BRAF", "PTEN", "NRAS"]),
    (
        "run a pathway analysis on EGFR, KRAS, BRAF, PTEN, NRAS, PIK3CA",
        ["EGFR", "KRAS", "BRAF", "PTEN", "NRAS", "PIK3CA"],
    ),
    (
        "Please run an enrichment on:\nEGFR\nKRAS\nBRAF\nPTEN\nNRAS\nAKT1",
        ["EGFR", "KRAS", "BRAF", "PTEN", "NRAS", "AKT1"],
    ),
    ("Analyse these:\nTP53\nMDM2\nCDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    (
        "Can you analyse these genes?\negfr\nkras\nbraf\npten",
        ["egfr", "kras", "braf", "pten"],
    ),
    ("run pathway analysis: egfr, kras, braf, pten", ["egfr", "kras", "braf", "pten"]),
    (
        "which pathways are my genes in? TP53, MDM2, CDKN1A, BAX",
        ["TP53", "MDM2", "CDKN1A", "BAX"],
    ),
    (
        "What pathways are over-represented in this list: TP53, MDM2, CDKN1A",
        ["TP53", "MDM2", "CDKN1A"],
    ),
    (
        "Here is my gene list, run it through Reactome: TP53 MDM2 CDKN1A BAX",
        ["TP53", "MDM2", "CDKN1A", "BAX"],
    ),
    (
        "TP53, MDM2, CDKN1A, BAX -- run an enrichment please",
        ["TP53", "MDM2", "CDKN1A", "BAX"],
    ),
    ("Find enriched pathways for TP53, MDM2, CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    ("analyze TP53, MDM2, CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    (
        "Can you run these through the Reactome analysis tool: TP53, MDM2, BAX",
        ["TP53", "MDM2", "BAX"],
    ),
    (
        "run ORA on my list: IL6 TNF IL1B CXCL8 CCL2",
        ["IL6", "TNF", "IL1B", "CXCL8", "CCL2"],
    ),
    (
        "Please analyse the following proteins: P04637, Q00987, P38936",
        ["P04637", "Q00987", "P38936"],
    ),
    ("do an analysis of these genes: egfr kras braf", ["egfr", "kras", "braf"]),
    (
        "I have a list of DE genes: MYC, CCND1, CDK4, E2F1. Which Reactome pathways are enriched?",
        ["MYC", "CCND1", "CDK4", "E2F1"],
    ),
    ("enrichment for egfr, kras, braf, pten please", ["egfr", "kras", "braf", "pten"]),
    (
        "can you do pathway enrichment on ENSG00000141510 ENSG00000135679",
        ["ENSG00000141510", "ENSG00000135679"],
    ),
    ("map these genes to pathways: TP53 MDM2 CDKN1A", ["TP53", "MDM2", "CDKN1A"]),
    (
        "Run an enrichment on GAPDH, ACTB, TUBB, VIM, KRT8, KRT18",
        ["GAPDH", "ACTB", "TUBB", "VIM", "KRT8", "KRT18"],
    ),
    (
        "run an ORA on CD4 CD8A CD3E CD19 MS4A1 NCAM1 ITGAM",
        ["CD4", "CD8A", "CD3E", "CD19", "MS4A1", "NCAM1", "ITGAM"],
    ),
    (
        "Run an ORA on EGFR, ERBB2, ERBB3, ERBB4, GRB2, SOS1, HRAS",
        ["EGFR", "ERBB2", "ERBB3", "ERBB4", "GRB2", "SOS1", "HRAS"],
    ),
    (
        "Perform a pathway analysis on mouse genes Trp53, Mdm2, Cdkn1a",
        ["Trp53", "Mdm2", "Cdkn1a"],
    ),
    (
        "please run a gene set enrichment on BRCA1 BRCA2 ATM ATR CHEK2",
        ["BRCA1", "BRCA2", "ATM", "ATR", "CHEK2"],
    ),
]

HELD_OUT_REQUESTS: list[tuple[str, list[str]]] = [
    (
        "can we do a gsa analysis in the chat. I want to do it with genes TP53, ERBB2 and RUNX2",
        ["TP53", "ERBB2", "RUNX2"],
    ),
    ("run a pathway analysis on TP53, ERBB2, RUNX2", ["TP53", "ERBB2", "RUNX2"]),
    (
        "Could you run an over-representation analysis for SOX9, RUNX2, SP7, COL1A1?",
        ["SOX9", "RUNX2", "SP7", "COL1A1"],
    ),
    (
        "Please perform an enrichment on these: NFKB1, RELA, IKBKB, CHUK",
        ["NFKB1", "RELA", "IKBKB", "CHUK"],
    ),
    (
        "I'd like to run ORA with STAT1, STAT2, IRF9, ISG15, MX1",
        ["STAT1", "STAT2", "IRF9", "ISG15", "MX1"],
    ),
    (
        "analyse my genes: pik3ca, akt1, mtor, rps6kb1",
        ["pik3ca", "akt1", "mtor", "rps6kb1"],
    ),
    (
        "run GSEA on\nCDK1\nCCNB1\nPLK1\nAURKA\nBUB1",
        ["CDK1", "CCNB1", "PLK1", "AURKA", "BUB1"],
    ),
    ("Do a Reactome enrichment for P04637 and P38398", ["P04637", "P38398"]),
    (
        "find pathways enriched in: SLC2A1, HK2, PFKP, LDHA, PKM",
        ["SLC2A1", "HK2", "PFKP", "LDHA", "PKM"],
    ),
    (
        "which pathways contain these genes? FGF2, FGFR1, FRS2, GRB2",
        ["FGF2", "FGFR1", "FRS2", "GRB2"],
    ),
    ("Run an analysis on insr, irs1, irs2, pik3r1", ["insr", "irs1", "irs2", "pik3r1"]),
    ("perform ORA: VEGFA; KDR; FLT1; NRP1", ["VEGFA", "KDR", "FLT1", "NRP1"]),
    (
        "please analyze CASP3, CASP8, CASP9 and APAF1",
        ["CASP3", "CASP8", "CASP9", "APAF1"],
    ),
    (
        "Can you run an enrichment analysis with my list of genes: Nanog, Pou5f1, Sox2, Klf4",
        ["Nanog", "Pou5f1", "Sox2", "Klf4"],
    ),
    (
        "submit TLR4, MYD88, TRAF6, IRAK4 for pathway analysis",
        ["TLR4", "MYD88", "TRAF6", "IRAK4"],
    ),
]

REVIEW_QUESTIONS: list[str] = [
    "Can you explain the enrichment of TP53 targets in apoptosis pathways?",
    "Is BRCA1 enriched in DNA repair pathways compared to BRCA2?",
    "Please explain how EGFR and KRAS signalling interact",
    "Can you tell me which pathways are enriched in my GSEA results for MYC and E2F targets?",
    "What does pathway analysis tell us about TP53 and MDM2?",
    "Why is my enrichment analysis showing ERBB2 and GRB7 together?",
    "I ran an ORA and got TP53 and CDKN1A at the top, what does that mean?",
    "Could you describe the role of SMAD2 and SMAD3 in TGF-beta signalling?",
    "Please summarise what is known about BRCA1 and BRCA2",
    "Can you do a literature summary on IL6 and STAT3?",
    "In a gene set analysis, why would HLA-A and B2M both show up?",
    "How does over-representation analysis handle genes like TP53 that are in many pathways?",
    "What pathways involve both PTEN and AKT1?",
    "Can I use UniProt IDs like P04637 and P38398 in the analysis tool?",
    "Is GSEA better than ORA for a list like CD4, CD8A, CD3E?",
    "Could you explain why IFNG and TNF are enriched in immune pathways?",
    "Please help me understand the interaction between MTOR and RPTOR",
    "What does a p<0.05 FDR mean for pathway analysis of NOTCH1 targets?",
    "Can you show me the reactions in which CDK1 and CCNB1 take part?",
    "I want to understand the enrichment of JAK2 and STAT5A in hematopoiesis",
    "Run me through how pathway analysis works for genes like APOE and APP",
    "Can you list the pathways where VEGFA and KDR are found?",
    "Would like to know if ESR1 and PGR are enriched in breast cancer pathways",
    "Help me interpret this: WNT signalling enriched, CTNNB1 and APC found",
    "Can you do a comparison of MAPK1 and MAPK3 functions?",
    "Please describe the reaction where ATP is hydrolysed by ABCB1",
    "How is NF-kB activated by TNF and IL1B?",
    "Which genes in the TCA cycle, like CS and IDH2, are enriched in my data?",
    "What is the enrichment score for TP53 in the Reactome GSA output?",
    "Could we discuss pathway analysis results where SOX2 and POU5F1 appear?",
    "I did an enrichment with DAVID for FOXO1 and FOXO3, can you compare with Reactome?",
    "Can you explain the difference between ORA and GSEA using BRCA1 and ATM as examples?",
    "Please tell me whether KRAS G12D is enriched in pancreatic cancer",
    "Can you check if my pathway analysis with PD-1 and CTLA4 is correct?",
    "Could you help me with an analysis of how TP53 and MDM2 regulate each other?",
    "Is there an enrichment of ERBB2 amplification in HER2+ breast cancer?",
    "Can you run through the steps of the insulin pathway: INS, INSR, IRS1?",
    "Do HIF1A and VHL appear together in any enriched pathway?",
    "What are the downstream effects of mTORC1 and mTORC2 in pathway analysis?",
    "Please explain why GAPDH and ACTB are used as housekeeping genes in enrichment",
    "Can you describe the ORA method? I have genes like TP53.",
    "How many genes are in the pathway enriched for BCL2 and BAX?",
    "Can we talk about the 2x enrichment of CD19 and MS4A1 in B cells?",
    "What would pathway analysis show for a knockout of IL2 and IL2RA?",
]

HELD_OUT_QUESTIONS: list[str] = [
    "Which pathways is TP53 in?",
    "Can you explain what an ORA does with genes like MYC and MAX?",
    "Why does my enrichment show HLA-A and HLA-B at the top?",
    "How are RUNX2 and SP7 related in osteoblast differentiation?",
    "Is ERBB2 over-represented in breast cancer compared to ERBB3?",
    "What is the role of NOTCH1 and JAG1 in development?",
    "Tell me about BRCA1 and BRCA2 in homologous recombination",
    "Can you summarise the MAPK cascade with BRAF and MEK1?",
    "I ran GSEA and CDK1 and PLK1 came up. Does that make sense?",
    "Please give me the reactions that involve ATM and CHEK2",
    "Could you find the literature on TP53 and MDM2?",
    "What does FDR mean in my analysis with EGFR and KRAS?",
    "Does CTNNB1 bind APC in the destruction complex?",
    "Show me the diagram for the pathway with SMAD2 and SMAD4",
    "What analysis can I do with DESeq2 output for 3 conditions?",
    "run the pathway browser for me",
    "Are IL6 and STAT3 enriched in the JAK-STAT pathway?",
    "Is there a pathway that links GLUT1 and HK2?",
    "Which of these involve TP53?",
    "Can you find pathways where both INS and INSR are present?",
]
